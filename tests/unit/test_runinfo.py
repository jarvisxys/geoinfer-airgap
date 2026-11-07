from __future__ import annotations

import json

import pytest

from geoinfer.utils import runinfo


def test_collect_is_json_serialisable() -> None:
    info = runinfo.collect()
    assert {"git_sha", "python", "packages", "torch"} <= info.keys()
    json.dumps(info)


def test_missing_package_is_none() -> None:
    assert runinfo.package_versions(("definitely-not-a-real-package-xyz",)) == {
        "definitely-not-a-real-package-xyz": None
    }


def test_missing_command_returns_none() -> None:
    assert runinfo._run(["definitely-not-a-real-command-xyz"]) is None


@pytest.mark.gpu
def test_torch_sees_gpu() -> None:
    info = runinfo.torch_info()
    assert info.get("cuda_available") is True
    assert len(info["capability"]) == 2
