import torch
from sklearn.cluster import MiniBatchKMeans
import time

print("Starting KMeans test...")
delta = torch.randn(2000)
values_np = delta.numpy().reshape(-1, 1)

t0 = time.time()
kmeans = MiniBatchKMeans(
    n_clusters=3,
    random_state=42,
    batch_size=1024,
    n_init="auto"
)
kmeans.fit(values_np)
print(f"Finished in {time.time() - t0:.4f} seconds!")
