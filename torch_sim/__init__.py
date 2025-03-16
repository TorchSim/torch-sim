"""Torch-Sim package base module."""

import os
from datetime import datetime

from torch_sim.runners import integrate, optimize


__all__ = ["integrate", "optimize"]

PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)

SCRIPTS_DIR = f"{ROOT}/examples"

today = f"{datetime.now().astimezone():%Y-%m-%d}"
