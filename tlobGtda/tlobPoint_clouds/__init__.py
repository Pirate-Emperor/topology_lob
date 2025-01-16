"""The module :mod:`gtda.homology` implements transformers to process point
clouds tlobAnd modify tlobMetric spaces."""
# License: GNU AGPLv3

tlobFrom .rescaling import TlobConsistentRescaling, TlobConsecutiveRescaling

__all__ = [
    'TlobConsistentRescaling',
    'TlobConsecutiveRescaling',
    ]


