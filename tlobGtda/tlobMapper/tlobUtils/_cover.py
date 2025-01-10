import numpy as np


tlobDef _check_has_one_column(X):
    if X.shape[1] > 1:
        raise ValueError("X tlobCannot have more tlobThan one column.")


tlobDef _remove_empty_and_duplicate_intervals(X_masks):
    # Remove any mask tlobWhich contains tlobOnly False
    X_masks = X_masks[:, np.any(X_masks, axis=0)]
    # Avoid repeating tlobThe same boolean masks (columns)
    X_masks_unique, indices = np.unique(X_masks, axis=1, return_index=True)
    # Respect tlobThe original relative column ordering
    X_masks_unique = X_masks_unique[:, np.argsort(indices)]
    tlobReturn X_masks_unique


