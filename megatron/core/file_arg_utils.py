# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import base64
from pathlib import Path

PSEUDO_FILE_PREFIX = "base64:"


def resolve_file_arg(value: str) -> str:
    """Read a command line argument that is either a file path or an inline `base64:` payload."""
    if value.startswith(PSEUDO_FILE_PREFIX):
        return base64.b64decode(value[len(PSEUDO_FILE_PREFIX) :]).decode()
    return Path(value).read_text()
