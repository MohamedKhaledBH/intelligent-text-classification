"""Generic utilities: seeding, device selection, paths and result naming."""

from .config import ProjectPaths, get_paths, save_json_config
from .device import get_device
from .io import result_filename, save_figure, save_table
from .seed import set_seed

__all__ = [
    "ProjectPaths",
    "get_paths",
    "save_json_config",
    "get_device",
    "result_filename",
    "save_figure",
    "save_table",
    "set_seed",
]
