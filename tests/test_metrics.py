"""
#############################################################################
### Evaluation metric unit tests
###
### @file test_metrics.py
### @author Sebastian Russo
### @date 2026
#############################################################################

MSE and relative L² on synthetic arrays.
"""

# Third-party imports
import numpy as np

# Project imports
from ffaoml.evaluation.metrics import mse, relative_l2

"""TESTS-----------------------------------------------------------"""


def test_mse_zero_on_identical() -> None:
    x = np.ones((2, 3, 4), dtype=np.float32)
    assert mse(x, x) == 0.0


def test_relative_l2_zero_on_identical() -> None:
    x = np.random.randn(4, 2, 5).astype(np.float32)
    assert relative_l2(x, x) == 0.0


def test_relative_l2_known_ratio() -> None:
    true = np.array([3.0, 4.0], dtype=np.float32)
    pred = np.array([0.0, 0.0], dtype=np.float32)
    assert relative_l2(pred, true) == 1.0
