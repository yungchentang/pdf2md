import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import date as Date
from pathlib import Path

from PIL import Image

import pdf2md
import pdf_to_markdown
from pdf_to_markdown import (
    ConversionTimeout,
    PdfJob,
    VisionTextBox,
    WatermarkSummary,
    build_glmocr_command,
    convert_pdf_job,
    discover_pdf_jobs,
    format_vision_observations,
    glmocr_executable,
    lookback_dates,
    macos_vision_ocr,
    markdown_has_meaningful_content,
    markdown_completed,
    normalize_ocr_markdown,
    preprocess_watermark_page,
    remaining_seconds,
    run_glmocr_cli,
)


class DiscoveryTest(unittest.TestCase):
    def test_all_scans_root_and_date_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            (source / "2026-04-28").mkdir(parents=True)
            (source / "not-a-date").mkdir()
            (source / "root.pdf").write_bytes(b"%PDF")
            (source / "2026-04-28" / "dated.pdf").write_bytes(b"%PDF")
            (source / "not-a-date" / "ignored.pdf").write_bytes(b"%PDF")

            jobs = discover_pdf_jobs(source, target, all_files=True)

            self.assertEqual([str(job.relative_path) for job in jobs], ["2026-04-28/dated.md", "root.md"])
            self.assertEqual(jobs[0].target_path, target.resolve() / "2026-04-28" / "dated.md")

    def test_lookback_scans_only_recent_date_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            for folder in ("2026-04-26", "2026-04-27", "2026-04-28", "2026-04-01"):
                (source / folder).mkdir(parents=True)
                (source / folder / f"{folder}.pdf").write_bytes(b"%PDF")
            (source / "root.pdf").write_bytes(b"%PDF")

            jobs = discover_pdf_jobs(
                source,
                target,
                lookback_days=3,
                today=Date(2026, 4, 28),
            )

            self.assertEqual(
                [str(job.relative_path) for job in jobs],
                ["2026-04-26/2026-04-26.md", "2026-04-27/2026-04-27.md", "2026-04-28/2026-04-28.md"],
            )

    def test_date_scans_only_specific_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            (source / "2026-04-28").mkdir(parents=True)
            (source / "2026-04-27").mkdir()
            (source / "2026-04-28" / "a.pdf").write_bytes(b"%PDF")
            (source / "2026-04-27" / "b.pdf").write_bytes(b"%PDF")

            jobs = discover_pdf_jobs(source, target, date="2026-04-28")

            self.assertEqual([str(job.relative_path) for job in jobs], ["2026-04-28/a.md"])

    def test_lookback_dates_includes_today(self):
        self.assertEqual(
            lookback_dates(3, today=Date(2026, 4, 28)),
            {"2026-04-28", "2026-04-27", "2026-04-26"},
        )


class CompletionTest(unittest.TestCase):
    def test_completed_frontmatter_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            path.write_text("---\nstatus: completed\n---\n\nThis file contains real OCR content.\n", encoding="utf-8")

            self.assertTrue(markdown_completed(path))

    def test_missing_or_failed_frontmatter_is_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            path.write_text("---\nstatus: failed\n---\n\nbody\n", encoding="utf-8")

            self.assertFalse(markdown_completed(path))
            self.assertFalse(markdown_completed(Path(tmp) / "missing.md"))

    def test_empty_code_fences_are_not_meaningful_or_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.md"
            path.write_text(
                "---\nstatus: completed\n---\n\n<!-- page 1 -->\n\n```markdown\n\n```\n",
                encoding="utf-8",
            )

            self.assertFalse(markdown_has_meaningful_content(path.read_text(encoding="utf-8")))
            self.assertFalse(markdown_completed(path))

    def test_normalize_ocr_markdown_unwraps_code_fence(self):
        self.assertEqual(normalize_ocr_markdown("```markdown\nhello\n```"), "hello")


class GlmOcrExecutableTest(unittest.TestCase):
    def test_finds_glmocr_from_virtualenv_prefix_even_when_python_is_symlinked(self):
        original_executable = sys.executable
        original_prefix = sys.prefix
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / ".venv-sdk"
            real_python = Path(tmp) / "homebrew" / "python3.14"
            venv_python = venv / "bin" / "python"
            glmocr = venv / "bin" / "glmocr"
            real_python.parent.mkdir(parents=True)
            venv_python.parent.mkdir(parents=True)
            real_python.write_text("", encoding="utf-8")
            venv_python.symlink_to(real_python)
            glmocr.write_text("", encoding="utf-8")
            sys.executable = str(venv_python)
            sys.prefix = str(venv)
            try:
                self.assertEqual(glmocr_executable(), str(glmocr))
            finally:
                sys.executable = original_executable
                sys.prefix = original_prefix

    def test_missing_glmocr_reports_setup_error(self):
        original_executable = sys.executable
        original_prefix = sys.prefix
        original_path = os.environ.get("PATH", "")
        with tempfile.TemporaryDirectory() as tmp:
            fake_python = Path(tmp) / "bin" / "python"
            fake_python.parent.mkdir()
            fake_python.write_text("", encoding="utf-8")
            sys.executable = str(fake_python)
            sys.prefix = str(Path(tmp) / ".venv-sdk")
            os.environ["PATH"] = ""
            try:
                with self.assertRaises(Exception) as context:
                    glmocr_executable()
                self.assertIn("Run `python3 pdf2md.py setup`", str(context.exception))
            finally:
                sys.executable = original_executable
                sys.prefix = original_prefix
                os.environ["PATH"] = original_path

    def test_glmocr_command_uses_packaged_config_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "page.png"
            output = Path(tmp) / "out"
            image.write_text("", encoding="utf-8")

            command = build_glmocr_command(
                image,
                None,
                output,
                api_port=8080,
                model="mlx-community/GLM-OCR-bf16",
            )

            self.assertIn("--mode", command)
            self.assertIn("selfhosted", command)
            self.assertIn("--layout-device", command)
            self.assertIn("cpu", command)
            self.assertIn("pipeline.ocr_api.api_path", command)
            self.assertIn("/chat/completions", command)
            self.assertNotIn("--config", command)


class TimeoutTest(unittest.TestCase):
    def test_remaining_seconds_raises_after_deadline(self):
        with self.assertRaises(ConversionTimeout):
            remaining_seconds(time.monotonic() - 1)


class WatermarkTest(unittest.TestCase):
    def test_preprocess_lightens_gray_watermark_and_preserves_black_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            output = Path(tmp) / "clean.png"
            image = Image.new("RGB", (2, 1))
            image.putdata([(20, 20, 20), (170, 172, 171)])
            image.save(source)

            ok, ratio = preprocess_watermark_page(source, output, max_changed_ratio=0.80)

            self.assertTrue(ok)
            self.assertGreater(ratio, 0)
            cleaned = Image.open(output).convert("RGB")
            pixel_source = getattr(cleaned, "get_flattened_data", cleaned.getdata)
            pixels = list(pixel_source())
            self.assertEqual(pixels[0], (20, 20, 20))
            self.assertGreater(pixels[1][0], 170)


class OcrFallbackTest(unittest.TestCase):
    def test_glmocr_no_markdown_output_allows_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "page.png"
            image.write_bytes(b"fake image")
            fake_glmocr = root / "fake-glmocr"
            fake_glmocr.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_glmocr.chmod(0o755)
            original_build = pdf_to_markdown.build_glmocr_command
            pdf_to_markdown.build_glmocr_command = lambda *args, **kwargs: [str(fake_glmocr)]

            try:
                text = run_glmocr_cli(
                    image,
                    None,
                    root / "out",
                    timeout_seconds=5,
                    api_port=8080,
                    model="mlx-community/GLM-OCR-bf16",
                )
            finally:
                pdf_to_markdown.build_glmocr_command = original_build

            self.assertEqual(text, "")

    def test_format_vision_observations_filters_metadata_and_watermarks(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "page.png"
            image = Image.new("RGB", (1000, 1000), "white")
            for x in range(100, 700):
                for y in range(240, 260):
                    image.putpixel((x, y), (10, 10, 10))
            image.save(image_path)
            boxes = [
                VisionTextBox("更新、LTA战略和股东回报等多个主题，这可能是股价的关键催化剂。最后，與KA", 0.10, 0.74, 0.70, 0.76, 0.5),
                VisionTextBox("won@jpmorgan.com 摩根士丹利证券（远东）", 0.69, 0.73, 0.93, 0.75, 0.5),
                VisionTextBox("李桑斯克（82-2）758 5146", 0.69, 0.70, 0.83, 0.72, 0.5),
                VisionTextBox("微信：Macro Guru", 0.30, 0.22, 0.42, 0.24, 0.5),
                VisionTextBox("3", 0.92, 0.03, 0.93, 0.05, 1.0),
            ]

            text = format_vision_observations(image_path, boxes)

            self.assertIn("关键催化剂。最后，", text)
            self.assertNotIn("與KA", text)
            self.assertNotIn("jpmorgan.com", text)
            self.assertNotIn("李桑斯克", text)
            self.assertNotIn("Macro", text)

    def test_macos_vision_ocr_runs_swift_script(self):
        original_swift = os.environ.get("PDF2MD_SWIFT")
        with tempfile.TemporaryDirectory() as tmp:
            fake_swift = Path(tmp) / "swift"
            script = Path(tmp) / "vision_ocr.swift"
            image = Path(tmp) / "page.png"
            fake_swift.write_text("#!/bin/sh\nprintf '這是一段可用的 OCR fallback 文字內容\\n'\n", encoding="utf-8")
            fake_swift.chmod(0o755)
            script.write_text("// fake script\n", encoding="utf-8")
            image.write_bytes(b"fake image")
            os.environ["PDF2MD_SWIFT"] = str(fake_swift)

            try:
                text = macos_vision_ocr(image, timeout_seconds=5, script_path=script)
            finally:
                if original_swift is None:
                    os.environ.pop("PDF2MD_SWIFT", None)
                else:
                    os.environ["PDF2MD_SWIFT"] = original_swift

            self.assertIn("OCR fallback", text)

    def test_convert_uses_macos_vision_when_mlx_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.pdf"
            target = root / "sample.md"
            image = root / "page.png"
            source.write_bytes(b"%PDF")
            image.write_bytes(b"fake image")
            job = PdfJob(source_path=source, target_path=target, relative_path=Path("sample.md"))
            original_prepare = pdf_to_markdown.prepare_page_images
            original_direct = pdf_to_markdown.direct_page_ocr
            original_vision = pdf_to_markdown.macos_vision_ocr
            pdf_to_markdown.prepare_page_images = lambda *args, **kwargs: ([image], WatermarkSummary(enabled=False))
            pdf_to_markdown.direct_page_ocr = lambda *args, **kwargs: ""
            pdf_to_markdown.macos_vision_ocr = lambda *args, **kwargs: "這是一段由 macOS Vision OCR fallback 產生的有效內容。"

            try:
                result = convert_pdf_job(
                    job,
                    work_root=root / "work",
                    ocr_runner=lambda *args, **kwargs: "```markdown\n\n```",
                )
            finally:
                pdf_to_markdown.prepare_page_images = original_prepare
                pdf_to_markdown.direct_page_ocr = original_direct
                pdf_to_markdown.macos_vision_ocr = original_vision

            output = target.read_text(encoding="utf-8")
            self.assertEqual(result.fallback_engines, ("macos_vision",))
            self.assertIn('fallback_ocr_engines: "macos_vision"', output)
            self.assertIn("macOS Vision OCR fallback", output)


class LaunchAgentTest(unittest.TestCase):
    def test_plist_contains_timeout_and_lookback(self):
        plist = pdf2md.build_launch_agent_plist(
            source=Path("/tmp/source"),
            target=Path("/tmp/target"),
            python_path=Path("/tmp/pdf2md/.venv-sdk/bin/python"),
            timeout_seconds=123,
            lookback_days=5,
        )

        args = plist["ProgramArguments"]
        self.assertIn("--timeout-seconds", args)
        self.assertIn("123", args)
        self.assertIn("--lookback-days", args)
        self.assertIn("5", args)
        self.assertNotIn("--all", args)
        self.assertEqual(plist["StartInterval"], pdf2md.DEFAULT_INTERVAL_SECONDS)

    def test_schedule_rejects_backfill_without_explicit_allow(self):
        args = argparse.Namespace(
            no_timeout=False,
            timeout_seconds=3000,
            all=True,
            lookback_days=3,
            allow_scheduled_backfill=False,
            date=None,
        )

        with self.assertRaises(SystemExit):
            pdf2md.install_schedule(args)

    def test_schedule_rejects_lookback_zero_without_explicit_allow(self):
        args = argparse.Namespace(
            no_timeout=False,
            timeout_seconds=3000,
            all=False,
            lookback_days=0,
            allow_scheduled_backfill=False,
            date=None,
        )

        with self.assertRaises(SystemExit):
            pdf2md.install_schedule(args)


class ServerLifecycleTest(unittest.TestCase):
    def test_server_startup_failure_stops_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake-server"
            script.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
            script.chmod(0o755)
            original_executable = pdf2md.mlx_server_executable
            pdf2md.mlx_server_executable = lambda: script
            server = pdf2md.MlxServer(port=65500, startup_timeout_seconds=1, log_dir=Path(tmp) / "logs")
            try:
                with self.assertRaises(Exception):
                    server.start()
                self.assertIsNone(server.process)
            finally:
                pdf2md.mlx_server_executable = original_executable

    def test_server_startup_error_includes_stderr_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake-server"
            script.write_text("#!/bin/sh\necho 'metal unavailable' >&2\nexit 1\n", encoding="utf-8")
            script.chmod(0o755)
            original_executable = pdf2md.mlx_server_executable
            pdf2md.mlx_server_executable = lambda: script
            server = pdf2md.MlxServer(port=65500, startup_timeout_seconds=3, log_dir=Path(tmp) / "logs")
            try:
                with self.assertRaises(Exception) as context:
                    server.start()
                self.assertIn("metal unavailable", str(context.exception))
            finally:
                pdf2md.mlx_server_executable = original_executable


class CliTest(unittest.TestCase):
    def test_run_args_allow_all_and_no_timeout(self):
        args = pdf2md.parse_args(
            [
                "run",
                "--source",
                "/tmp/source",
                "--target",
                "/tmp/target",
                "--all",
                "--no-timeout",
            ]
        )

        self.assertTrue(args.all)
        self.assertTrue(args.no_timeout)

    def test_run_args_reject_no_timeout_without_full_scan(self):
        args = pdf2md.parse_args(
            [
                "run",
                "--source",
                "/tmp/source",
                "--target",
                "/tmp/target",
                "--no-timeout",
            ]
        )

        with self.assertRaises(SystemExit):
            pdf2md.validate_run_args(args)


class RunStatusTest(unittest.TestCase):
    def test_dry_run_writes_finished_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            status_file = Path(tmp) / "status.json"
            source.mkdir()
            (source / "sample.pdf").write_bytes(b"%PDF")
            args = pdf2md.parse_args(
                [
                    "run",
                    "--source",
                    str(source),
                    "--target",
                    str(target),
                    "--all",
                    "--dry-run",
                    "--status-file",
                    str(status_file),
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                rc = pdf2md.run_once(args)
            status = json.loads(status_file.read_text(encoding="utf-8"))

            self.assertEqual(rc, 0)
            self.assertEqual(status["state"], "finished")
            self.assertEqual(status["phase"], "dry-run-complete")
            self.assertEqual(status["stats"]["discovered"], 1)


if __name__ == "__main__":
    unittest.main()
