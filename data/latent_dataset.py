"""Dataset for loading pre-extracted latent tokens."""

import os

import torch
from torch.utils.data import Dataset


STEM_CLASSES = {'drums': 0, 'bass': 1, 'other': 2, 'vocals': 3}


class LatentDataset(Dataset):
    """Loads cached latent tokens from extract_latents.py output.

    Args:
        latent_dir: Directory containing latents_{split}.pt files
        split: 'train', 'val', or 'test'
        normalize: If True, normalize latents to zero mean, unit variance
                   using dataset statistics saved during extraction
    """

    def __init__(self, latent_dir, split='train', normalize=True):
        path = os.path.join(latent_dir, f'latents_{split}.pt')
        data = torch.load(path, weights_only=True)

        self.latents = data['latents']  # [N, S, D]

        if normalize:
            self.mean = data['mean']  # [S, D]
            self.std = data['std'].clamp(min=1e-6)  # [S, D]
            self.latents = (self.latents - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        # Convert stem names to class indices
        stems = data.get('stems', [])
        if stems:
            self.classes = torch.tensor(
                [STEM_CLASSES.get(s, 2) for s in stems], dtype=torch.long
            )
        else:
            self.classes = torch.zeros(len(self.latents), dtype=torch.long)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.classes[idx]

    def denormalize(self, z):
        """Convert normalized latents back to original scale for decoding."""
        if self.mean is not None:
            return z * self.std.to(z.device) + self.mean.to(z.device)
        return z
