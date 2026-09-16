from subprocess import CompletedProcess

import pytest

from zeek_repro.data import require_java_17


def test_require_java_17_accepts_supported_runtime(monkeypatch):
    monkeypatch.setattr(
        "zeek_repro.data.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "", 'openjdk version "17.0.16"'),
    )
    require_java_17()


def test_require_java_17_rejects_java_25(monkeypatch):
    monkeypatch.setattr(
        "zeek_repro.data.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "", 'openjdk version "25.0.4"'),
    )
    with pytest.raises(RuntimeError, match=r"JDK 17.*detected Java 25"):
        require_java_17()
