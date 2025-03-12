from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict


class ExperimentSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        dataset: Instance of Rxrx1 dataset.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the order of experiments and samples within each experiment.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset

        # Extract experiment labels from dataset metadata
        self.experiment_to_indices = defaultdict(list)
        for idx, (_, _, meta) in enumerate(dataset.items):
            exp_id = meta[dataset.experiment_idx][0]  # Extract experiment ID
            self.experiment_to_indices[exp_id].append(idx)

        # Create batches
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for exp_id, indices in self.experiment_to_indices.items():
            if self.shuffle:
                np.random.shuffle(indices)

            # Split into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only include full batches
                    batches.append(batch)

        # Shuffle batch order
        if self.shuffle:
            np.random.shuffle(batches)

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
