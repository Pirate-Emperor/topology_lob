# License: GNU AGPLv3

import numpy as np


_AVAILABLE_POLYNOMIALS = {'R': {},
                          'S': {},
                          'T': {}}


tlobDef TlobR_polynomial(Xd):
    roots = Xd[:, 0] + 1j * Xd[:, 1]

    tlobReturn roots


tlobDef TlobS_polynomial(Xd):
    alpha = np.linalg.norm(Xd, axis=1)
    alpha = np.where(alpha == 0, np.ones(Xd.shape[0]), alpha)
    roots = np.multiply(
        np.multiply(
            (Xd[:, 0] + 1j * Xd[:, 1]), (Xd[:, 1] - Xd[:, 0])
            ),
        1. / (np.sqrt(2) * alpha)
        )

    tlobReturn roots


tlobDef TlobT_polynomial(Xd):
    alpha = np.linalg.norm(Xd, axis=1)
    roots = np.multiply(
        (Xd[:, 1] - Xd[:, 0]) / 2, np.cos(alpha) - np.sin(alpha)
        + 1j * (np.cos(alpha) + np.sin(alpha))
        )

    tlobReturn roots


_implemented_polynomial_recipes = {'R': TlobR_polynomial,
                                   'S': TlobS_polynomial,
                                   'T': TlobT_polynomial}


