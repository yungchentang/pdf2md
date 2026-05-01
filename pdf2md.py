#!/usr/bin/env python3
"""Unified entrypoint for local GLM-OCR PDF-to-Markdown conversion."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import queue
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pdf_to_markdown import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MODEL,
    DEFAULT_STATUS_FILE,
    DEFAULT_TIMEOUT_SECONDS,
    ConversionTimeout,
    Pdf2MdError,
    convert_pdf_job,
    discover_pdf_jobs,
    markdown_completed,
    write_status,
)


DEFAULT_LABEL = "com.kumi.pdf2md"
DEFAULT_INTERVAL_SECONDS = 3600
DEFAULT_PORT = 8080
DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS = 900
PROGRESS_HEARTBEAT_SECONDS = 30
LAUNCHD_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def project_root() -> Path:
    return Path(__file__).resolve().parent


def sdk_python(root: Path | None = None) -> Path:
    root = root or project_root()
    candidate = root / ".venv-sdk" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable).resolve()


def mlx_python(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / ".venv-mlx" / "bin" / "python"


def launch_agent_path(label: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_recent_log(path: Path, max_bytes: int = 80_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        if path.stat().st_size > max_bytes:
            fh.seek(-max_bytes, os.SEEK_END)
        return fh.read().decode("utf-8", errors="replace").replace("\r", "\n")


def print_recent_lines(title: str, path: Path, lines: int) -> None:
    print(f"\n{title}: {path}")
    text = read_recent_log(path)
    recent = [line for line in text.splitlines() if line.strip()][-lines:]
    if not recent:
        print("(empty)")
        return
    for line in recent:
        print(line)


def run_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def parse_launchctl_print(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("state = "):
            fields["state"] = stripped.removeprefix("state = ")
        if stripped.startswith("pid = "):
            fields["pid"] = stripped.removeprefix("pid = ")
        if stripped.startswith("last exit code = "):
            fields["last_exit"] = stripped.removeprefix("last exit code = ")
    return fields


def create_venv(venv_dir: Path) -> Path:
    python = venv_dir / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=project_root(), check=True)
        print(f"created venv: {venv_dir}")
    else:
        print(f"venv already exists: {venv_dir}")
    return python


def install_requirements(python: Path, requirements: Path) -> None:
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=project_root(), check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], cwd=project_root(), check=True)


def mlx_server_executable(root: Path | None = None) -> Path:
    root = root or project_root()
    executable = root / ".venv-mlx" / "bin" / "mlx_vlm.server"
    if executable.exists():
        return executable
    raise Pdf2MdError(f"mlx-vlm executable not found: {executable}; run `python3 pdf2md.py setup` first")


def health_check(port: int, *, model: str, timeout_seconds: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        if data.get("status") == "healthy":
            loaded_model = data.get("loaded_model")
            if not loaded_model or str(loaded_model) == model:
                return True
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "max_tokens": 10,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False
    return "choices" in data


@dataclass
class MlxServer(AbstractContextManager["MlxServer"]):
    port: int = DEFAULT_PORT
    model: str = DEFAULT_MODEL
    startup_timeout_seconds: int = DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS
    log_dir: Path = project_root() / "logs"
    process: subprocess.Popen[str] | None = None
    stdout_fh: Any = None
    stderr_fh: Any = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    status_callback: Callable[[str], None] | None = None

    def __enter__(self) -> "MlxServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        executable = mlx_server_executable()
        self.stdout_path = self.log_dir / "mlx-vlm.out.log"
        self.stderr_path = self.log_dir / "mlx-vlm.err.log"
        self.stdout_fh = self.stdout_path.open("ab")
        self.stderr_fh = self.stderr_path.open("ab")
        command = [
            str(executable),
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--model",
            self.model,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=project_root(),
            stdout=self.stdout_fh,
            stderr=self.stderr_fh,
            text=True,
            start_new_session=True,
        )
        try:
            self.wait_until_ready()
        except Exception:
            self.stop()
            raise

    def wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.startup_timeout_seconds
        started_at = time.monotonic()
        next_heartbeat = started_at + PROGRESS_HEARTBEAT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise Pdf2MdError(
                    f"mlx-vlm server exited early with code {self.process.returncode}\n"
                    f"{self.recent_server_logs()}"
                )
            request_timeout = min(30, max(1, int(deadline - time.monotonic())))
            if health_check(self.port, model=self.model, timeout_seconds=request_timeout):
                if self.status_callback:
                    self.status_callback(f"mlx-vlm server ready after {int(time.monotonic() - started_at)}s")
                return
            if self.status_callback and time.monotonic() >= next_heartbeat:
                elapsed = int(time.monotonic() - started_at)
                remaining = max(0, int(deadline - time.monotonic()))
                self.status_callback(f"still starting mlx-vlm server; elapsed={elapsed}s, remaining_timeout={remaining}s")
                next_heartbeat = time.monotonic() + PROGRESS_HEARTBEAT_SECONDS
            time.sleep(2)
        raise Pdf2MdError(
            f"mlx-vlm server did not become healthy within {self.startup_timeout_seconds} seconds\n"
            f"{self.recent_server_logs()}"
        )

    def recent_server_logs(self, lines: int = 30) -> str:
        parts: list[str] = []
        for title, path in (("stdout", self.stdout_path), ("stderr", self.stderr_path)):
            if not path or not path.exists():
                continue
            text = read_recent_log(path, max_bytes=40_000)
            recent = [line for line in text.splitlines() if line.strip()][-lines:]
            if recent:
                parts.append(f"{title} tail:\n" + "\n".join(recent))
        return "\n\n".join(parts) if parts else "server logs are empty"

    def stop(self) -> None:
        process = self.process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        for handle in (self.stdout_fh, self.stderr_fh):
            if handle:
                handle.close()
        self.process = None


@dataclass(frozen=True)
class JobOutcome:
    index: int
    relative_path: str
    output_path: str | None = None
    error: str | None = None


def setup_environment(args: argparse.Namespace) -> int:
    root = project_root()
    mlx_venv = root / ".venv-mlx"
    sdk_venv = root / ".venv-sdk"
    mlx_py = create_venv(mlx_venv)
    sdk_py = create_venv(sdk_venv)
    if not args.no_install:
        install_requirements(mlx_py, root / "requirements-mlx.txt")
        install_requirements(sdk_py, root / "requirements-sdk.txt")
        print("installed requirements")
    if not args.skip_model_warmup:
        print("warming up GLM-OCR model through mlx-vlm; first run can take a while")
        with MlxServer(
            port=args.port,
            model=args.model,
            startup_timeout_seconds=args.warmup_timeout_seconds,
            log_dir=root / "logs",
        ):
            print(f"model server warmed up on port {args.port}")
    return 0


def validate_run_args(args: argparse.Namespace) -> None:
    if args.all and args.date:
        raise SystemExit("--all cannot be combined with --date")
    if args.no_timeout and not (args.all or args.lookback_days == 0):
        raise SystemExit("--no-timeout is only allowed with --all or --lookback-days 0")
    if args.timeout_seconds <= 0 and not args.no_timeout:
        raise SystemExit("--timeout-seconds must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")


def format_progress(index: int, total: int, label: str, *, width: int = 24) -> str:
    if total <= 0:
        return f"progress [{' ' * width}] 0/0 {label}"
    done = max(0, min(index, total))
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(100 * done / total)
    return f"progress [{bar}] {done}/{total} ({percent:3d}%) {label}"


def format_total_progress(completed: int, total: int, stats: dict[str, int]) -> str:
    label = f"done={completed}, processed={stats.get('processed', 0)}, failed={stats.get('failed', 0)}"
    return format_progress(completed, total, label)


def process_job(
    *,
    index: int,
    job: Any,
    args: argparse.Namespace,
    work_root: Path,
    progress_events: queue.Queue[dict[str, Any]] | None = None,
) -> JobOutcome:
    def emit(stage: str, page: int, total_pages: int) -> None:
        if progress_events is None:
            return
        progress_events.put(
            {
                "index": index,
                "file": str(job.relative_path),
                "stage": stage,
                "page": page,
                "total_pages": total_pages,
            }
        )

    try:
        file_deadline = None if args.no_timeout else time.monotonic() + args.timeout_seconds
        result = convert_pdf_job(
            job,
            work_root=work_root,
            config_path=None,
            model=args.model,
            api_port=args.port,
            preprocess_watermark=args.preprocess_watermark,
            timeout_seconds=None if args.no_timeout else args.timeout_seconds,
            deadline=file_deadline,
            include_header=args.include_header,
            progress_callback=emit,
        )
        return JobOutcome(index=index, relative_path=str(job.relative_path), output_path=str(result.output_path))
    except Exception as exc:
        return JobOutcome(index=index, relative_path=str(job.relative_path), error=str(exc))


def drain_progress_events(progress_events: queue.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(progress_events.get_nowait())
        except queue.Empty:
            return events


def format_page_progress(event: dict[str, Any]) -> str:
    file = event.get("file", "(unknown)")
    stage = event.get("stage", "progress")
    page = int(event.get("page", 0))
    total_pages = int(event.get("total_pages", 0))
    if stage == "pages-ready":
        return f"worker {event.get('index')}: {file} has {total_pages} pages"
    if total_pages > 0 and page > 0:
        return f"worker {event.get('index')}: {file} {stage} page {page}/{total_pages}"
    return f"worker {event.get('index')}: {file} {stage}"


def run_once(args: argparse.Namespace) -> int:
    validate_run_args(args)
    root = project_root()
    source = Path(args.source).expanduser().resolve()
    target = Path(args.target).expanduser().resolve()
    status_file = Path(args.status_file).expanduser().resolve() if args.status_file else None
    all_files = bool(args.all or args.lookback_days == 0)
    stats = {"discovered": 0, "skipped": 0, "processed": 0, "failed": 0}
    outputs: list[str] = []
    failures: list[dict[str, str]] = []

    write_status(
        status_file,
        {
            "state": "running",
            "phase": "scanning",
            "source": str(source),
            "target": str(target),
            "all": all_files,
            "date": args.date,
            "lookback_days": args.lookback_days,
            "timeout_seconds": None if args.no_timeout else args.timeout_seconds,
            "workers": args.workers,
            "stats": stats,
        },
    )
    try:
        jobs = discover_pdf_jobs(
            source,
            target,
            all_files=all_files,
            date=args.date,
            lookback_days=args.lookback_days,
        )
        stats["discovered"] = len(jobs)
        pending = []
        for job in jobs:
            if not args.force and markdown_completed(job.target_path, allow_plain=not args.include_header):
                stats["skipped"] += 1
                if args.dry_run:
                    print(f"skip {job.relative_path}: already completed")
                continue
            pending.append(job)
            if args.dry_run:
                print(f"pending {job.relative_path}")
        print(
            "scan: "
            f"discovered={stats['discovered']}, skipped={stats['skipped']}, "
            f"pending={len(pending)}, workers={args.workers}"
        )

        write_status(
            status_file,
            {
                "state": "finished" if args.dry_run or not pending else "running",
                "phase": "dry-run-complete" if args.dry_run else ("done" if not pending else "ready"),
                "source": str(source),
                "target": str(target),
                "total_to_process": len(pending),
                "pending_files": [str(job.relative_path) for job in pending],
                "server_pid": None,
                "workers": args.workers,
                "stats": stats,
            },
        )
        if args.dry_run or not pending:
            print("summary: " + ", ".join(f"{key}={value}" for key, value in stats.items()))
            return 0

        work_root = root / ".cache" / "work"

        write_status(
            status_file,
            {
                "state": "running",
                "phase": "starting-server",
                "source": str(source),
                "target": str(target),
                "total_to_process": len(pending),
                "workers": args.workers,
                "stats": stats,
            },
        )
        print(f"starting mlx-vlm server on port {args.port}; model={args.model}")
        with MlxServer(
            port=args.port,
            model=args.model,
            startup_timeout_seconds=args.server_startup_timeout_seconds,
            log_dir=root / "logs",
            status_callback=lambda message: print(f"server: {message}"),
        ) as server:
            completed_count = 0
            progress_events: queue.Queue[dict[str, Any]] = queue.Queue()
            active_pages: dict[str, dict[str, Any]] = {}
            last_heartbeat = time.monotonic()
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures: set[Future[JobOutcome]] = set()
                next_job_index = 0

                def submit_next() -> None:
                    nonlocal next_job_index
                    if next_job_index >= len(pending):
                        return
                    job = pending[next_job_index]
                    index = next_job_index + 1
                    next_job_index += 1
                    print(f"start {index}/{len(pending)}: {job.relative_path}")
                    futures.add(
                        executor.submit(
                            process_job,
                            index=index,
                            job=job,
                            args=args,
                            work_root=work_root,
                            progress_events=progress_events,
                        )
                    )

                for _ in range(min(args.workers, len(pending))):
                    submit_next()

                while futures:
                    done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for event in drain_progress_events(progress_events):
                        print(format_page_progress(event))
                        if event.get("stage") in {"page-start", "page-done", "pages-ready"}:
                            active_pages[str(event.get("file"))] = event

                    now = time.monotonic()
                    if not done and now - last_heartbeat >= PROGRESS_HEARTBEAT_SECONDS:
                        active = ", ".join(
                            f"{Path(file).name}: page {event.get('page')}/{event.get('total_pages')}"
                            for file, event in sorted(active_pages.items())
                            if int(event.get("total_pages", 0)) > 0
                        )
                        suffix = f" active: {active}" if active else " waiting for OCR"
                        print(format_total_progress(completed_count, len(pending), stats) + f"; running={len(futures)};{suffix}")
                        last_heartbeat = now

                    for future in done:
                        outcome = future.result()
                        completed_count += 1
                        active_pages.pop(outcome.relative_path, None)
                        if outcome.error is None:
                            stats["processed"] += 1
                            assert outcome.output_path is not None
                            outputs.append(outcome.output_path)
                            print(format_total_progress(completed_count, len(pending), stats))
                            print(f"processed {outcome.relative_path}: {outcome.output_path}")
                        else:
                            stats["failed"] += 1
                            failures.append({"file": outcome.relative_path, "error": outcome.error})
                            print(format_total_progress(completed_count, len(pending), stats))
                            print(f"failed {outcome.relative_path}: {outcome.error}", file=sys.stderr)
                        submit_next()
                        active = max(0, len(futures))
                        write_status(
                            status_file,
                            {
                                "state": "running",
                                "phase": "processing",
                                "source": str(source),
                                "target": str(target),
                                "completed_count": completed_count,
                                "active_jobs": active,
                                "active_pages": active_pages,
                                "total_to_process": len(pending),
                                "server_pid": server.pid,
                                "workers": args.workers,
                                "stats": stats,
                            },
                        )

        print("summary: " + ", ".join(f"{key}={value}" for key, value in stats.items()))
        write_status(
            status_file,
            {
                "state": "finished",
                "phase": "done" if not stats["failed"] else "done-with-failures",
                "source": str(source),
                "target": str(target),
                "server_pid": None,
                "workers": args.workers,
                "stats": stats,
                "outputs": outputs,
                "failures": failures,
            },
        )
        return 1 if stats["failed"] else 0
    except ConversionTimeout as exc:
        stats["failed"] += 1
        print(f"timed out: {exc}", file=sys.stderr)
        write_status(
            status_file,
            {
                "state": "finished",
                "phase": "timed-out",
                "source": str(source),
                "target": str(target),
                "server_pid": None,
                "timeout_seconds": None if args.no_timeout else args.timeout_seconds,
                "workers": args.workers,
                "error_message": str(exc),
                "stats": stats,
            },
        )
        return 124
    except KeyboardInterrupt:
        write_status(
            status_file,
            {
                "state": "finished",
                "phase": "interrupted",
                "source": str(source),
                "target": str(target),
                "server_pid": None,
                "workers": args.workers,
                "stats": stats,
            },
        )
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        stats["failed"] += 1
        write_status(
            status_file,
            {
                "state": "finished",
                "phase": "failed",
                "source": str(source),
                "target": str(target),
                "server_pid": None,
                "workers": args.workers,
                "error_message": str(exc),
                "stats": stats,
            },
        )
        print(f"failed: {exc}", file=sys.stderr)
        return 1


def build_launch_agent_plist(
    *,
    source: Path,
    target: Path,
    python_path: Path,
    label: str = DEFAULT_LABEL,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    all_files: bool = False,
    date: str | None = None,
    force: bool = False,
    preprocess_watermark: bool = True,
    include_header: bool = False,
    workers: int = 1,
    port: int = DEFAULT_PORT,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    root = project_root()
    log_dir = (log_dir or (root / "logs")).expanduser().resolve()
    args = [
        str(python_path.expanduser().resolve()),
        str(root / "pdf2md.py"),
        "run",
        "--source",
        str(source.expanduser().resolve()),
        "--target",
        str(target.expanduser().resolve()),
        "--timeout-seconds",
        str(timeout_seconds),
        "--status-file",
        str(root / DEFAULT_STATUS_FILE),
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]
    if all_files:
        args.append("--all")
    elif date:
        args.extend(["--date", date])
    else:
        args.extend(["--lookback-days", str(lookback_days)])
    if force:
        args.append("--force")
    if not preprocess_watermark:
        args.append("--no-preprocess-watermark")
    if include_header:
        args.append("--header")
    return {
        "Label": label,
        "ProgramArguments": args,
        "WorkingDirectory": str(root),
        "StartInterval": int(interval_seconds),
        "RunAtLoad": False,
        "StandardOutPath": str(log_dir / "launchd.out.log"),
        "StandardErrorPath": str(log_dir / "launchd.err.log"),
        "EnvironmentVariables": {"PATH": LAUNCHD_PATH},
    }


def write_launch_agent(plist: dict[str, Any], label: str) -> Path:
    path = launch_agent_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(plist["StandardOutPath"]).parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        plistlib.dump(plist, fh, sort_keys=False)
    return path


def install_schedule(args: argparse.Namespace) -> int:
    if args.no_timeout:
        raise SystemExit("scheduled runs must use --timeout-seconds; --no-timeout is not allowed")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive for scheduled runs")
    scheduled_backfill = args.all or args.lookback_days == 0
    if scheduled_backfill and not args.allow_scheduled_backfill:
        raise SystemExit("scheduled backfill is blocked unless --allow-scheduled-backfill is provided")
    if args.date and args.all:
        raise SystemExit("--date cannot be combined with --all")
    python_path = Path(args.python).expanduser() if args.python else sdk_python()
    if not python_path.exists():
        raise SystemExit(f"python executable does not exist: {python_path}; run `python3 pdf2md.py setup` first")
    plist = build_launch_agent_plist(
        source=Path(args.source),
        target=Path(args.target),
        python_path=python_path,
        label=args.label,
        interval_seconds=args.interval_seconds,
        timeout_seconds=args.timeout_seconds,
        lookback_days=args.lookback_days,
        all_files=args.all,
        date=args.date,
        force=args.force,
        preprocess_watermark=args.preprocess_watermark,
        include_header=args.include_header,
        workers=args.workers,
        port=args.port,
        log_dir=Path(args.log_dir).expanduser().resolve() if args.log_dir else None,
    )
    if args.dry_run:
        plistlib.dump(plist, sys.stdout.buffer, sort_keys=False)
        return 0
    path = write_launch_agent(plist, args.label)
    print(f"wrote LaunchAgent: {path}")
    print(f"source: {Path(args.source).expanduser().resolve()}")
    print(f"target: {Path(args.target).expanduser().resolve()}")
    print(f"timeout_seconds: {args.timeout_seconds}")
    if args.load:
        gui_target = f"gui/{os.getuid()}"
        run_launchctl(["bootout", gui_target, str(path)])
        bootstrap = run_launchctl(["bootstrap", gui_target, str(path)])
        if bootstrap.returncode != 0:
            print(bootstrap.stdout, end="")
            print(bootstrap.stderr, end="", file=sys.stderr)
            return bootstrap.returncode
        print(f"loaded LaunchAgent into {gui_target}")
        if args.start_now:
            kickstart = run_launchctl(["kickstart", "-k", f"{gui_target}/{args.label}"])
            if kickstart.returncode != 0:
                print(kickstart.stdout, end="")
                print(kickstart.stderr, end="", file=sys.stderr)
                return kickstart.returncode
            print("started job now")
    else:
        print(f"load with: launchctl bootstrap gui/$(id -u) {path}")
    return 0


def show_status(args: argparse.Namespace) -> int:
    root = project_root()
    print(f"project: {root}")
    print(f"sdk_python: {sdk_python(root)} ({'exists' if sdk_python(root).exists() else 'missing'})")
    print(f"mlx_python: {mlx_python(root)} ({'exists' if mlx_python(root).exists() else 'missing'})")
    print(f"mlx_server: {root / '.venv-mlx' / 'bin' / 'mlx_vlm.server'}")

    plist_path = launch_agent_path(args.label)
    plist: dict[str, Any] | None = None
    if plist_path.exists():
        with plist_path.open("rb") as fh:
            plist = plistlib.load(fh)
    print(f"label: {args.label}")
    print(f"plist: {plist_path} ({'exists' if plist else 'missing'})")
    if plist:
        launchctl = run_launchctl(["print", f"gui/{os.getuid()}/{args.label}"])
        loaded = launchctl.returncode == 0
        fields = parse_launchctl_print(launchctl.stdout)
        print(f"loaded: {loaded}")
        print(f"state: {fields.get('state', 'unloaded' if not loaded else 'unknown')}")
        if "pid" in fields:
            print(f"pid: {fields['pid']}")
        if "last_exit" in fields:
            print(f"last_exit: {fields['last_exit']}")
        print(f"start_interval_seconds: {plist.get('StartInterval', '(missing)')}")
        program_args = [str(item) for item in plist.get("ProgramArguments", [])]
        print(f"command: {' '.join(program_args)}")

    status_path = Path(args.status_file).expanduser().resolve() if args.status_file else root / DEFAULT_STATUS_FILE
    status = load_json(status_path)
    print(f"status_file: {status_path} ({'exists' if status else 'missing'})")
    if status:
        stats = status.get("stats", {})
        print(f"run_state: {status.get('state', 'unknown')}")
        print(f"phase: {status.get('phase', 'unknown')}")
        print(f"updated_at: {status.get('updated_at', '(missing)')}")
        print(
            "run_counts: "
            f"discovered={stats.get('discovered', 0)}, skipped={stats.get('skipped', 0)}, "
            f"processed={stats.get('processed', 0)}, failed={stats.get('failed', 0)}"
        )
        if status.get("current_file"):
            print(f"current_file: {status.get('current_file')}")
        if status.get("server_pid"):
            print(f"server_pid: {status.get('server_pid')}")
        if status.get("error_message"):
            print(f"error: {status.get('error_message')}")
    if args.logs:
        print_recent_lines("launchd stdout", root / "logs" / "launchd.out.log", args.lines)
        print_recent_lines("launchd stderr", root / "logs" / "launchd.err.log", args.lines)
        print_recent_lines("mlx-vlm stdout", root / "logs" / "mlx-vlm.out.log", args.lines)
        print_recent_lines("mlx-vlm stderr", root / "logs" / "mlx-vlm.err.log", args.lines)
    return 0


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="Source folder containing PDFs or YYYY-MM-DD folders.")
    parser.add_argument("--target", required=True, help="Target folder for Markdown outputs.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--all", action="store_true", help="Scan all root PDFs and all YYYY-MM-DD folders.")
    parser.add_argument("--date", help="Scan only one YYYY-MM-DD source folder.")
    parser.add_argument("--force", action="store_true", help="Reprocess completed Markdown outputs.")
    parser.add_argument("--preprocess-watermark", dest="preprocess_watermark", action="store_true", default=True)
    parser.add_argument("--no-preprocess-watermark", dest="preprocess_watermark", action="store_false")
    parser.add_argument("--header", dest="include_header", action="store_true", default=False, help="Include YAML frontmatter and generated title in Markdown outputs.")
    parser.add_argument("--no-header", dest="include_header", action="store_false", help="Write only converted content without YAML frontmatter or title.")
    parser.add_argument("--workers", type=int, default=1, help="Number of PDF files to process concurrently.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Create venvs, install dependencies, and warm up the model.")
    setup_parser.add_argument("--no-install", action="store_true")
    setup_parser.add_argument("--skip-model-warmup", action="store_true")
    setup_parser.add_argument("--warmup-timeout-seconds", type=int, default=DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS)
    setup_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    setup_parser.add_argument("--model", default=DEFAULT_MODEL)
    setup_parser.set_defaults(func=setup_environment)

    run_parser = subparsers.add_parser("run", help="Convert queued PDF files once.")
    add_common_run_args(run_parser)
    run_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    run_parser.add_argument("--no-timeout", action="store_true")
    run_parser.add_argument("--server-startup-timeout-seconds", type=int, default=DEFAULT_SERVER_STARTUP_TIMEOUT_SECONDS)
    run_parser.add_argument("--status-file", default=str(project_root() / DEFAULT_STATUS_FILE))
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--dry-run", action="store_true", help="Scan and show pending files without starting OCR.")
    run_parser.set_defaults(func=run_once)

    install_parser = subparsers.add_parser("install-schedule", help="Install a per-user LaunchAgent.")
    add_common_run_args(install_parser)
    install_parser.add_argument("--label", default=DEFAULT_LABEL)
    install_parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    install_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    install_parser.add_argument("--no-timeout", action="store_true")
    install_parser.add_argument("--python", help="Python executable. Defaults to .venv-sdk/bin/python.")
    install_parser.add_argument("--log-dir", help="LaunchAgent log directory. Defaults to ./logs.")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--load", action="store_true")
    install_parser.add_argument("--start-now", action="store_true")
    install_parser.add_argument("--allow-scheduled-backfill", action="store_true")
    install_parser.set_defaults(func=install_schedule)

    status_parser = subparsers.add_parser("status", help="Show env, LaunchAgent, and recent run status.")
    status_parser.add_argument("--label", default=DEFAULT_LABEL)
    status_parser.add_argument("--status-file", default=str(project_root() / DEFAULT_STATUS_FILE))
    status_parser.add_argument("--logs", action="store_true")
    status_parser.add_argument("--lines", type=int, default=20)
    status_parser.set_defaults(func=show_status)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
