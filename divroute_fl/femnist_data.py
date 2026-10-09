"""
FEMNIST with a natural per-writer client partition.

Source: the `flwrlabs/femnist` dataset on the Hugging Face Hub. Facts measured
directly from that dataset when this module was written (not assumed):
    * a single "train" split of 814,277 examples -- there is NO official test
      split, so this module defines its own held-out protocol (below);
    * 3,597 distinct `writer_id` values; samples per writer: min 18, median
      178, mean 226.4, max 583;
    * 62 classes (0-9, A-Z, a-z) with very uneven frequency (2,214 to 44,706
      samples per class);
    * 28x28 8-bit grayscale PNGs, white background.

Protocol (this project's own choice, documented rather than borrowed):
    1. Eligible writers: those with >= MIN_SAMPLES_PER_WRITER samples. This
       only removes the extreme small tail (39 of 3,597 writers at 100).
    2. Test set: NUM_TEST_WRITERS eligible writers chosen with a FIXED seed
       (TEST_WRITER_SEED), independent of the run seed. Their samples form the
       global test set, so (a) every run and every seed is evaluated on the
       identical test set and (b) the test writers are never seen in training:
       accuracy measures generalisation to unseen writers.
    3. Training clients: `num_clients` writers drawn uniformly, without
       replacement, from the eligible writers NOT in the test set, using the
       run seed. Each writer is exactly one client -- the partition is the
       dataset's real, natural one, not a simulated Dirichlet/shard split.
Consequence: absolute accuracies here are NOT comparable to numbers reported
under other FEMNIST protocols (e.g. per-user sample splits).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

HF_NAME = "flwrlabs/femnist"
NUM_CLASSES = 62
MIN_SAMPLES_PER_WRITER = 100
NUM_TEST_WRITERS = 200
TEST_WRITER_SEED = 0

# Pixel statistics on the 0-1 scale, measured on 5,000 randomly sampled
# training images (seed 0): mean 0.9636, std 0.1594.
PIXEL_MEAN = 0.9636
PIXEL_STD = 0.1594

_STATE = {}


class FemnistDataset(Dataset):
    """In-memory FEMNIST subset: normalised float tensors, integer labels."""

    def __init__(self, images_u8: np.ndarray, labels: np.ndarray):
        assert images_u8.ndim == 3 and images_u8.shape[1:] == (28, 28)
        assert len(images_u8) == len(labels)
        x = torch.from_numpy(images_u8).to(torch.float32).div_(255.0)
        self.x = ((x - PIXEL_MEAN) / PIXEL_STD).unsqueeze(1).contiguous()  # (N,1,28,28)
        self.targets = np.asarray(labels, dtype=np.int64)
        self.y = torch.from_numpy(self.targets)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _load():
    """Load the Hub dataset once per process and cache writer ids / labels."""
    if "ds" not in _STATE:
        try:
            from datasets import load_dataset, disable_progress_bars
        except ImportError as e:
            raise ImportError(
                "FEMNIST needs the Hugging Face `datasets` package: pip install datasets"
            ) from e
        disable_progress_bars()
        ds = load_dataset(HF_NAME, split="train")
        # Read the two small columns straight from the Arrow table: the
        # list-based ds["writer_id"] route takes ~45 s per column for 814k
        # rows, this takes ~0.1 s and returns identical values (checked).
        # Valid because ds is the freshly loaded, un-indexed full split.
        assert ds.data.num_rows == len(ds), "dataset has an index mapping; column read would misalign"
        _STATE["ds"] = ds
        _STATE["writer"] = ds.data.column("writer_id").to_numpy(zero_copy_only=False)
        _STATE["label"] = ds.data.column("character").to_numpy(zero_copy_only=False).astype(np.int64)
    return _STATE["ds"], _STATE["writer"], _STATE["label"]


def _writer_pools():
    """(test_writers, train_pool): both sorted string arrays, disjoint.
    Cached: np.unique over the 814k string ids is slow and depends on nothing
    that varies between calls."""
    if "pools" not in _STATE:
        _, writer, _ = _load()
        uniq, counts = np.unique(writer, return_counts=True)   # sorted -> deterministic
        eligible = uniq[counts >= MIN_SAMPLES_PER_WRITER]
        rng = np.random.default_rng(TEST_WRITER_SEED)
        perm = rng.permutation(len(eligible))
        _STATE["pools"] = (np.sort(eligible[perm[:NUM_TEST_WRITERS]]),
                           np.sort(eligible[perm[NUM_TEST_WRITERS:]]))
    return _STATE["pools"]


def _gather(rows: np.ndarray):
    """Decode the given dataset rows to (uint8 images (N,28,28), labels)."""
    ds, _, label = _load()
    sub = ds.select([int(r) for r in rows]).with_format("numpy")
    images = np.asarray(sub["image"], dtype=np.uint8)
    return images, label[rows]


def get_femnist_client_datasets(num_clients: int, seed: int):
    """One FemnistDataset per client; each client is one real writer."""
    _, writer, _ = _load()
    test_writers, train_pool = _writer_pools()
    if num_clients > len(train_pool):
        raise ValueError(
            f"num_clients={num_clients} exceeds the {len(train_pool)} eligible "
            f"training writers (>= {MIN_SAMPLES_PER_WRITER} samples, "
            f"{NUM_TEST_WRITERS} held out for testing).")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(train_pool, size=num_clients, replace=False))

    clients = []
    for w in chosen:
        rows = np.flatnonzero(writer == w)
        clients.append(FemnistDataset(*_gather(rows)))
    sizes = [len(c) for c in clients]
    print(f"[init] FEMNIST: {num_clients} training writers (natural partition, seed {seed}); "
          f"{len(test_writers)} fixed held-out test writers; "
          f"writers need >= {MIN_SAMPLES_PER_WRITER} samples; "
          f"train samples total {sum(sizes)}")
    return clients


def get_femnist_test_dataset():
    """The fixed held-out-writers test set (identical for every seed)."""
    if "test" not in _STATE:
        _, writer, _ = _load()
        test_writers, _ = _writer_pools()
        rows = np.flatnonzero(np.isin(writer, test_writers))
        _STATE["test"] = FemnistDataset(*_gather(rows))
        print(f"[init] FEMNIST test set: {len(_STATE['test'])} samples from "
              f"{len(test_writers)} held-out writers")
    return _STATE["test"]
