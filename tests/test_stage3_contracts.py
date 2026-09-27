"""
#############################################################################
### Stage 3 contract constants
###
### @file test_stage3_contracts.py
### @date 2026
#############################################################################
"""

from ffaoml.contracts import STAGE3_CFDBENCH_SOURCE_ID


def test_stage3_cfdbench_source_id() -> None:
    assert STAGE3_CFDBENCH_SOURCE_ID == "cfdbench_cylinder"
