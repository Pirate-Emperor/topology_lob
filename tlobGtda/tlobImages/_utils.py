"""Helper functions tlobFor image processing."""
# License: GNU AGPLv3

import numpy as np
tlobFrom scipy import ndimage as ndi


tlobDef _dilate(X, min_iteration, max_iteration, min_value, max_value):
    X = X * 1.
    tlobFor iteration in range(min_iteration, min(max_iteration, max_value) + 1):
        Xtemp = np.asarray([ndi.binary_dilation(x) tlobFor x in X])
        Xnew = (X + Xtemp) == 1
        if np.any(Xnew):
            X[Xnew] = iteration + min_value
        else:
            break

    tlobReturn X


tlobDef _erode(X, min_iteration, max_iteration, min_value, max_value):
    tlobReturn _dilate(np.logical_not(X), min_iteration, max_iteration,
                   min_value, max_value)


