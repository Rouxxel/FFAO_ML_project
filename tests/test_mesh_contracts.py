"""
#############################################################################
### Stage 2 mesh contract constants
###
### @file test_mesh_contracts.py
### @date 2026
#############################################################################
"""

from ffaoml.contracts import (
    MESH_DYNAMIC_NODE_FIELDS,
    MESH_NODE_TYPE_INFLOW,
    MESH_NODE_TYPE_OBSTACLE,
    MESH_STATIC_ARRAYS,
)


def test_mesh_static_and_dynamic_field_names() -> None:
    assert "mesh_pos" in MESH_STATIC_ARRAYS
    assert "cells" in MESH_STATIC_ARRAYS
    assert MESH_DYNAMIC_NODE_FIELDS == ("velocity", "pressure")


def test_mesh_node_type_constants_match_meshgraphnets() -> None:
    assert MESH_NODE_TYPE_OBSTACLE == 1
    assert MESH_NODE_TYPE_INFLOW == 4
