DivRoute-FL Lite: How the simulation actually works


The Big Picture

At its core, this project is a Federated Averaging (FedAvg) sandbox built with PyTorch. We are simulating a central server and 20 different clients collaboratively training a CNN on CIFAR-10. The key is that raw data never leaves the clients; they only share model weights. This codebase is the "Person A" baseline—the plumbing is all there, but the advanced research bits (like the divergence scoring and compression) are still placeholders for now.


The Architecture

1. main.py: The entry point that glues everything together.
2. config.py: A single dataclass where we keep all the hyperparameters like learning rates and communication rounds.
3. data.py: Handles the CIFAR-10 download and the Dirichlet partitioning.
4. model.py: A lightweight CNN with about 200k parameters. It’s small enough to run on a standard CPU.
5. server.py & client.py: These handle the actual federated logic—aggregation on the server side and local SGD on the client side.
6. logger.py: Saves everything to a JSON file after every round so we don't lose progress if it crashes.


Data: The Dirichlet Split

To make this a realistic FL simulation, we can't just give everyone an even split of the data. In data.py, we use a Dirichlet distribution with an alpha-value of 0.5.This creates a "non-IID" environment, meaning some clients might end up with 3,000 images of mostly airplanes and ships, while others might only get a few hundred cats. This is what makes federated learning hard—the models have to learn from very different, biased datasets.


The Training Loop

When you run the simulation, it follows a specific cycle for each of the 100 rounds:
1. Selection: The server picks 10 random clients to participate this round.
2. Local Training: Each selected client makes a "deep copy" of the global model. This is important because they need to train on their own data without mutating the server's master model weights. They run 3 epochs of SGD and then pass their new weights back to the server.
3. Aggregation: The server takes all those client models and performs FedAvg. It's a weighted average, so a client with 3,000 images has more influence on the new global model than a client with 500 images.
4. The "Delta": The server calculates the difference between the new global model and the old one (the "delta"). Eventually, we’ll use this to send compressed updates back to the clients.
5. Evaluation: The updated global model is tested against a separate 10,000-image test set to see if the accuracy is actually improving.


What’s still a "Work in Progress"

Since this is a research prototype, we have a few "stubs" in server.py that still need to be built out:
1. Divergence Scoring: Currently just returns 1.0. Later, we’ll use cosine similarity to see how much a client’s model has "drifted" from the global one.
2. Tier Assignment: Right now everyone is Tier 1. The goal is to sort clients into tiers based on that drift.
3. Compression: Currently, we send the full model delta (~800KB) to every client. We’ll eventually implement top-k sparsification to cut down on that bandwidth.


Why we designed it this way

1. Reproducibility: We seeded everything (NumPy, Torch, Random) so that if you run it twice, you get the exact same results.
2. Efficiency: The SimpleCNN is intentionally small. We want to spend our time testing the FL logic, not waiting hours for a huge model to train.
3. Safety: Flushing the logs to run.json every single round is a bit slower, but it’s a lifesaver if your laptop dies halfway through a 100-round run.