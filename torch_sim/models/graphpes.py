"""Deprecated module for importing GraphPESWrapper, AtomicGraph, and GraphPESModel.

This module is deprecated. Please use the ts.models.graphpes_framework module instead.
"""

import warnings

from .graphpes_framework import AtomicGraph, GraphPESModel, GraphPESWrapper


warnings.warn(
    "Importing from the ts.models.graphpes module is deprecated. "
    "Please use the ts.models.graphpes_framework module instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AtomicGraph",
    "GraphPESModel",
    "GraphPESWrapper",
]
