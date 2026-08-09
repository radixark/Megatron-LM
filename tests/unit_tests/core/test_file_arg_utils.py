# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import base64

import pytest

from megatron.core.file_arg_utils import PSEUDO_FILE_PREFIX, resolve_file_arg


class TestResolveFileArg:
    def test_reads_a_plain_file_path(self, tmp_path):
        """A bare path keeps working, so existing launchers are unaffected."""
        path = tmp_path / "recipe.yaml"
        path.write_text("configs:\n")

        assert resolve_file_arg(str(path)) == "configs:\n"

    def test_decodes_an_inline_base64_payload(self):
        """The inline form needs no filesystem shared with the job launcher."""
        encoded = base64.b64encode(b"configs:\n").decode()

        assert resolve_file_arg(f"{PSEUDO_FILE_PREFIX}{encoded}") == "configs:\n"

    def test_round_trips_multiline_utf8_content(self):
        """Recipes are multi-line and may carry non-ascii comments."""
        text = "configs:\n  bf16:\n    # 中文注释\n"
        encoded = base64.b64encode(text.encode()).decode()

        assert resolve_file_arg(f"{PSEUDO_FILE_PREFIX}{encoded}") == text

    def test_a_missing_path_still_raises(self, tmp_path):
        """A typo must fail loudly rather than silently yield an empty recipe."""
        with pytest.raises(FileNotFoundError):
            resolve_file_arg(str(tmp_path / "absent.yaml"))
