"""Tests for the API and job runner.

These use a throwaway workspace so they never touch real runs. Nothing here starts
actual training — what is tested is the contract the GUI depends on.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Point every path helper at a temporary directory for the duration of a test."""
    monkeypatch.setenv("LLMFORGE_HOME", str(tmp_path / "ws"))
    yield tmp_path


@pytest.fixture
def client():
    from llmforge.api.main import app

    return TestClient(app)


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    for i in range(3):
        (folder / f"doc_{i}.txt").write_text(
            f"Document {i}. " + "The quick brown fox jumps over the lazy dog. " * 40
        )
    return folder


# --------------------------------------------------------------------------
# system endpoints
# --------------------------------------------------------------------------


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["version"]


def test_browse_lists_only_directories(client, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a_file.txt").write_text("x")

    body = client.get("/api/browse", params={"path": str(tmp_path)}).json()
    names = [e["name"] for e in body["entries"]]
    assert "sub" in names
    assert "a_file.txt" not in names


def test_browse_hides_dotfiles(client, tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "visible").mkdir()

    names = [
        e["name"] for e in client.get("/api/browse", params={"path": str(tmp_path)}).json()["entries"]
    ]
    assert names == ["visible"]


def test_browse_counts_usable_files(client, corpus):
    body = client.get("/api/browse", params={"path": str(corpus.parent)}).json()
    entry = next(e for e in body["entries"] if e["name"] == "corpus")
    assert entry["n_files"] == 3


def test_browse_reports_parent(client, tmp_path):
    (tmp_path / "sub").mkdir()
    body = client.get("/api/browse", params={"path": str(tmp_path / "sub")}).json()
    assert body["parent"] == str(tmp_path)


def test_browse_missing_directory_is_404(client, tmp_path):
    assert client.get("/api/browse", params={"path": str(tmp_path / "nope")}).status_code == 404


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------


def test_runs_empty_initially(client):
    assert client.get("/api/runs").json() == []


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_start_rejects_a_bad_folder(client, tmp_path):
    response = client.post(
        "/api/runs", json={"spec": {"folder": str(tmp_path / "missing")}, "name": None}
    )
    assert response.status_code == 400


def test_cancelling_an_unknown_run_is_404(client):
    assert client.post("/api/runs/nope/cancel").status_code == 404


def test_analyze_rejects_a_bad_folder(client, tmp_path):
    response = client.post("/api/analyze", json={"folder": str(tmp_path / "missing")})
    assert response.status_code in (400, 404)


def test_analyze_rejects_an_empty_folder(client, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert client.post("/api/analyze", json={"folder": str(empty)}).status_code == 400


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


def test_spec_round_trips(tmp_path):
    from llmforge.jobs.spec import RunSpec

    spec = RunSpec(folder="/data", base="org/model", method="lora", seed=7)
    spec.write(tmp_path)
    assert RunSpec.read(tmp_path) == spec


def test_spec_knows_which_mode_it_is():
    from llmforge.jobs.spec import RunSpec

    assert RunSpec(folder="/d", base="org/m").is_finetune
    assert not RunSpec(folder="/d").is_finetune


def test_analyze_request_carries_overrides_into_the_spec():
    from llmforge.jobs.spec import AnalyzeRequest

    spec = AnalyzeRequest(folder="/d", base="org/m", method="qlora", epochs=5).to_spec("run")
    assert (spec.base, spec.method, spec.epochs, spec.name) == ("org/m", "qlora", 5, "run")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def test_reconcile_marks_dead_runs_failed():
    from llmforge.core import registry
    from llmforge.jobs import runner

    registry.create(run_id="ghost", mode="pretrain", plan={})
    registry.update("ghost", status="running")

    assert "ghost" in runner.reconcile()
    assert registry.get("ghost").status == "failed"


def test_reconcile_leaves_finished_runs_alone():
    from llmforge.core import registry
    from llmforge.jobs import runner

    registry.create(run_id="done", mode="pretrain", plan={})
    registry.update("done", status="completed")

    runner.reconcile()
    assert registry.get("done").status == "completed"


def test_a_live_process_counts_as_alive():
    from llmforge.core import paths, registry
    from llmforge.jobs import runner

    registry.create(run_id="live", mode="pretrain", plan={})
    # Our own pid is certainly running.
    (paths.run_dir("live") / "worker.pid").write_text(str(os.getpid()))
    assert runner.is_alive("live")


def test_a_missing_pid_file_is_not_alive():
    from llmforge.core import registry
    from llmforge.jobs import runner

    registry.create(run_id="nopid", mode="pretrain", plan={})
    assert not runner.is_alive("nopid")


def test_metrics_are_read_incrementally():
    from llmforge.core import paths, registry
    from llmforge.jobs import runner

    registry.create(run_id="m", mode="pretrain", plan={})
    path = paths.run_dir("m") / "metrics.jsonl"
    path.write_text(
        "".join(json.dumps({"step": i, "loss": 1.0 / (i + 1)}) + "\n" for i in range(3))
    )

    first, offset = runner.read_metrics("m")
    assert len(first) == 3 and offset == 3

    with path.open("a") as handle:
        handle.write(json.dumps({"step": 3, "loss": 0.2}) + "\n")

    more, offset = runner.read_metrics("m", offset)
    assert len(more) == 1 and more[0]["step"] == 3 and offset == 4


def test_partial_metric_lines_are_skipped():
    """A worker mid-write must not crash the reader."""
    from llmforge.core import paths, registry
    from llmforge.jobs import runner

    registry.create(run_id="partial", mode="pretrain", plan={})
    (paths.run_dir("partial") / "metrics.jsonl").write_text(
        json.dumps({"step": 1}) + "\n" + '{"step": 2, "lo'
    )
    records, _ = runner.read_metrics("partial")
    assert len(records) == 1


def test_samples_are_returned_newest_first():
    from llmforge.core import paths, registry
    from llmforge.jobs import runner

    registry.create(run_id="s", mode="pretrain", plan={})
    directory = paths.run_dir("s") / "samples"
    directory.mkdir()
    for step in (10, 200, 30):
        (directory / f"step_{step:06d}.txt").write_text(f"sample at {step}")

    assert [s["step"] for s in runner.samples("s")] == [200, 30, 10]


# --------------------------------------------------------------------------
# desktop launcher
# --------------------------------------------------------------------------


def test_port_selection_skips_a_busy_port():
    import socket

    from llmforge import app as launcher

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        busy = held.getsockname()[1]

        port, serving = launcher.choose_port(busy)
        # Something is listening but it is not us, so move along.
        assert port != busy
        assert serving is False


def test_desktop_entry_is_written(monkeypatch, tmp_path):
    from llmforge import app as launcher

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    entry, icon = launcher.install_desktop_entry()
    contents = entry.read_text()

    assert entry.exists() and icon.exists()
    assert "Type=Application" in contents
    assert "Terminal=false" in contents
    # More than one main category makes the launcher appear twice in the menu.
    assert contents.count("Categories=Development;\n") == 1


def test_desktop_entry_can_be_removed(monkeypatch, tmp_path):
    from llmforge import app as launcher

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    launcher.install_desktop_entry()

    removed = launcher.uninstall_desktop_entry()
    assert len(removed) == 2
    assert not any(p.exists() for p in removed)


def test_a_partial_final_line_is_retried_not_skipped():
    """The offset must not step over a line the worker was still writing."""
    from llmforge.core import paths, registry
    from llmforge.jobs import runner

    registry.create(run_id="retry", mode="pretrain", plan={})
    path = paths.run_dir("retry") / "metrics.jsonl"
    path.write_text(json.dumps({"step": 1}) + "\n" + '{"step": 2, "lo')

    records, offset = runner.read_metrics("retry")
    assert len(records) == 1 and offset == 1

    # The worker finishes the line; the next poll should deliver it exactly once.
    path.write_text(json.dumps({"step": 1}) + "\n" + json.dumps({"step": 2}) + "\n")
    more, offset = runner.read_metrics("retry", offset)
    assert [r["step"] for r in more] == [2] and offset == 2


# --------------------------------------------------------------------------
# portability: diagnosing a machine this was not built on
# --------------------------------------------------------------------------


def test_missing_driver_is_named_as_such(monkeypatch):
    """The three causes of "no GPU" need different remedies, so they need different
    messages. A wheel/driver mismatch installs cleanly and fails at the first CUDA
    call, which is the most confusing way this goes wrong on a new machine."""
    from llmforge import doctor

    monkeypatch.setattr(doctor, "nvidia_driver_version", lambda: None)
    assert "no NVIDIA driver" in doctor.diagnose_missing_cuda()


def test_cpu_only_wheel_is_named_as_such(monkeypatch):
    import torch

    from llmforge import doctor

    monkeypatch.setattr(doctor, "nvidia_driver_version", lambda: "550.54.15")
    monkeypatch.setattr(torch.version, "cuda", None)
    message = doctor.diagnose_missing_cuda()
    assert "CPU-only build" in message
    assert "index-url" in message


def test_driver_too_old_names_the_versions_and_the_fix(monkeypatch):
    import torch

    from llmforge import doctor

    # CUDA 13 needs driver 580+; this machine has a 12.x-era driver.
    monkeypatch.setattr(doctor, "nvidia_driver_version", lambda: "535.104.05")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    message = doctor.diagnose_missing_cuda()
    assert "580" in message and "535.104.05" in message
    assert "cu126" in message, "should suggest a build this driver can run"


def test_compatible_versions_point_elsewhere(monkeypatch):
    """When driver and wheel agree, the cause is something else — usually a
    CUDA_VISIBLE_DEVICES that hides everything."""
    import torch

    from llmforge import doctor

    monkeypatch.setattr(doctor, "nvidia_driver_version", lambda: "580.159.03")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    assert "CUDA_VISIBLE_DEVICES" in doctor.diagnose_missing_cuda()


def test_minimum_drivers_are_ordered():
    from llmforge.doctor import MIN_DRIVER

    assert MIN_DRIVER[13] > MIN_DRIVER[12]
