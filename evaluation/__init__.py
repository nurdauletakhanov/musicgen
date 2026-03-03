"""Evaluation module for music generation autoencoder."""

# Principled checkpoint selection result.
# Selected by minimizing mean |MixRate - 1.0| across alphas on the test set.
# See results/checkpoint_selection.json for full comparison.
# Re-derive: python -m evaluation.select_checkpoint
SELECTED_EPOCH = 104
