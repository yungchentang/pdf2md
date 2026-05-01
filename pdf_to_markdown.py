#!/usr/bin/env python3
"""PDF discovery and GLM-OCR conversion helpers for pdf2md."""

from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib import request as urlrequest
from urllib.error import URLError


DEFAULT_MODEL = "mlx-community/GLM-OCR-bf16"
DEFAULT_LOOKBACK_DAYS = 3
DEFAULT_TIMEOUT_SECONDS = 3000
DEFAULT_STATUS_FILE = Path(".cache/status.json")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Pdf2MdError(Exception):
    """Expected operational error."""


class ConversionTimeout(Pdf2MdError):
    """The current run exceeded its timeout budget."""


@dataclass(frozen=True)
class PdfJob:
    source_path: Path
    target_path: Path
    relative_path: Path
    date_folder: str | None = None

    @property
    def job_id(self) -> str:
        digest = hashlib.sha256(str(self.source_path).encode("utf-8")).hexdigest()[:16]
        return f"pdf-{digest}"


@dataclass(frozen=True)
class WatermarkSummary:
    enabled: bool
    applied_pages: int = 0
    fallback_pages: int = 0
    changed_ratios: tuple[float, ...] = ()

    @property
    def mode(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.fallback_pages and not self.applied_pages:
            return "fallback"
        if self.fallback_pages:
            return "partial"
        return "applied"


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    page_count: int
    watermark: WatermarkSummary
    fallback_engines: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisionTextBox:
    text: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    confidence: float = 0.0

    @property
    def mid_y(self) -> float:
        return (self.min_y + self.max_y) / 2.0


def now_local() -> str:
    return datetime.now().replace(microsecond=0).astimezone().isoformat()


def parse_date_folder(name: str) -> Date | None:
    if not DATE_RE.fullmatch(name):
        return None
    try:
        return Date.fromisoformat(name)
    except ValueError:
        return None


def lookback_dates(days: int, today: Date | None = None) -> set[str]:
    if days <= 0:
        return set()
    today = today or Date.today()
    return {(today - timedelta(days=offset)).isoformat() for offset in range(days)}


def iter_pdf_files(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def target_for_pdf(source_root: Path, target_root: Path, pdf_path: Path) -> tuple[Path, Path, str | None]:
    relative = pdf_path.relative_to(source_root)
    date_folder = relative.parts[0] if len(relative.parts) == 2 and parse_date_folder(relative.parts[0]) else None
    target_relative = relative.with_suffix(".md")
    return target_root / target_relative, target_relative, date_folder


def discover_pdf_jobs(
    source_root: Path,
    target_root: Path,
    *,
    all_files: bool = False,
    date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    today: Date | None = None,
) -> list[PdfJob]:
    source_root = source_root.expanduser().resolve()
    target_root = target_root.expanduser().resolve()
    if not source_root.exists():
        raise Pdf2MdError(f"source folder does not exist: {source_root}")
    if date and all_files:
        raise Pdf2MdError("--date cannot be combined with --all")
    if date and parse_date_folder(date) is None:
        raise Pdf2MdError(f"--date must use YYYY-MM-DD: {date}")

    pdfs: list[Path] = []
    if date:
        pdfs.extend(iter_pdf_files(source_root / date))
    elif all_files or lookback_days <= 0:
        pdfs.extend(iter_pdf_files(source_root))
        for child in sorted(source_root.iterdir()):
            if child.is_dir() and parse_date_folder(child.name):
                pdfs.extend(iter_pdf_files(child))
    else:
        for folder in sorted(lookback_dates(lookback_days, today=today)):
            pdfs.extend(iter_pdf_files(source_root / folder))

    jobs: list[PdfJob] = []
    for pdf_path in sorted(pdfs):
        target_path, relative_path, date_folder = target_for_pdf(source_root, target_root, pdf_path)
        jobs.append(PdfJob(source_path=pdf_path, target_path=target_path, relative_path=relative_path, date_folder=date_folder))
    return jobs


def read_text(path: Path, max_bytes: int = 200_000) -> str:
    with path.open("rb") as fh:
        return fh.read(max_bytes).decode("utf-8", errors="replace")


def markdown_completed(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    text = read_text(path, max_bytes=20_000)
    if not text.startswith("---\n"):
        return False
    end = text.find("\n---", 4)
    if end == -1:
        return False
    frontmatter = text[4:end]
    return bool(re.search(r"(?m)^status:\s*[\"']?completed[\"']?\s*$", frontmatter)) and markdown_has_meaningful_content(text)


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    return text[end + 4 :]


def normalize_ocr_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    fence = re.fullmatch(r"```(?:markdown)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text


def markdown_has_meaningful_content(text: str, *, min_chars: int = 20) -> bool:
    body = strip_frontmatter(text)
    body = re.sub(r"\A\s*# .*(?:\n|$)", "", body, count=1)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
    body = re.sub(r"```(?:markdown)?\s*```", "", body, flags=re.IGNORECASE)
    body = normalize_ocr_markdown(body)
    visible = re.sub(r"[\s#`*_>\-[\]().,:;!，。；：、（）]+", "", body)
    return len(visible) >= min_chars


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return '""'
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_status(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    status = dict(payload)
    status.setdefault("schema_version", "1.0")
    status["updated_at"] = now_local()
    try:
        write_json(path, status)
    except OSError as exc:
        print(f"warning: failed to write status file {path}: {exc}", file=sys.stderr)


def remaining_seconds(deadline: float | None) -> int | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ConversionTimeout("run timed out")
    return max(1, int(remaining))


def extract_page_images(pdf_path: Path, output_dir: Path, *, dpi: int = 220) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return extract_page_images_pymupdf(pdf_path, output_dir, dpi=dpi)
    except ImportError:
        return extract_page_images_pypdf(pdf_path, output_dir)


def extract_page_images_pymupdf(pdf_path: Path, output_dir: Path, *, dpi: int) -> list[Path]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        raise
    images: list[Path] = []
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    with fitz.open(str(pdf_path)) as doc:
        for index, page in enumerate(doc, start=1):
            output = output_dir / f"page-{index:04d}.png"
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(output))
            images.append(output)
    if not images:
        raise Pdf2MdError(f"PDF has no pages: {pdf_path}")
    return images


def extract_page_images_pypdf(pdf_path: Path, output_dir: Path) -> list[Path]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise Pdf2MdError("Install PyMuPDF or pypdf to extract PDF pages") from exc

    reader = PdfReader(str(pdf_path))
    images: list[Path] = []
    for index, page in enumerate(reader.pages, start=1):
        page_images = list(getattr(page, "images", []))
        if not page_images:
            raise Pdf2MdError(
                f"Page {index} cannot be rendered without PyMuPDF and has no embedded image fallback: {pdf_path}"
            )
        largest = max(page_images, key=lambda item: getattr(item.image, "width", 0) * getattr(item.image, "height", 0))
        output = output_dir / f"page-{index:04d}.png"
        largest.image.save(output)
        images.append(output)
    if not images:
        raise Pdf2MdError(f"PDF has no pages: {pdf_path}")
    return images


def preprocess_watermark_page(source: Path, output: Path, *, max_changed_ratio: float = 0.30) -> tuple[bool, float]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise Pdf2MdError("Pillow is required for watermark preprocessing") from exc

    with Image.open(source) as image:
        rgb = image.convert("RGB")
    pixel_source = getattr(rgb, "get_flattened_data", rgb.getdata)
    pixels = list(pixel_source())
    changed = 0
    cleaned: list[tuple[int, int, int]] = []
    for r, g, b in pixels:
        hi = max(r, g, b)
        lo = min(r, g, b)
        brightness = (r + g + b) / 3.0
        is_gray = hi - lo <= 18
        is_watermark_like = is_gray and 115 <= brightness <= 235
        if is_watermark_like:
            changed += 1
            cleaned.append(
                (
                    min(255, int(r + (255 - r) * 0.70)),
                    min(255, int(g + (255 - g) * 0.70)),
                    min(255, int(b + (255 - b) * 0.70)),
                )
            )
        else:
            cleaned.append((r, g, b))
    changed_ratio = changed / max(len(pixels), 1)
    if changed_ratio <= 0 or changed_ratio > max_changed_ratio:
        return False, changed_ratio
    rgb.putdata(cleaned)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(output)
    return True, changed_ratio


def prepare_page_images(
    pdf_path: Path,
    work_dir: Path,
    *,
    preprocess_watermark: bool,
) -> tuple[list[Path], WatermarkSummary]:
    raw_dir = work_dir / "pages-raw"
    images = extract_page_images(pdf_path, raw_dir)
    if not preprocess_watermark:
        return images, WatermarkSummary(enabled=False)

    cleaned_dir = work_dir / "pages-cleaned"
    final_images: list[Path] = []
    applied = 0
    fallback = 0
    ratios: list[float] = []
    for image in images:
        output = cleaned_dir / image.name
        try:
            ok, ratio = preprocess_watermark_page(image, output)
        except Exception as exc:
            print(f"warning: watermark preprocessing failed for {image.name}: {exc}", file=sys.stderr)
            ok, ratio = False, 0.0
        ratios.append(ratio)
        if ok:
            applied += 1
            final_images.append(output)
        else:
            fallback += 1
            final_images.append(image)
    return final_images, WatermarkSummary(
        enabled=True,
        applied_pages=applied,
        fallback_pages=fallback,
        changed_ratios=tuple(round(value, 6) for value in ratios),
    )


def glmocr_executable() -> str:
    candidates = [
        Path(sys.prefix) / "bin" / "glmocr",
        Path(sys.executable).parent / "glmocr",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    found = shutil.which("glmocr")
    if found:
        return found
    expected = candidates[0]
    raise Pdf2MdError(
        f"glmocr executable not found in SDK environment: {expected}. "
        "Run `python3 pdf2md.py setup` and execute with `.venv-sdk/bin/python`."
    )


def newest_markdown_file(directory: Path) -> Path | None:
    candidates = sorted(directory.rglob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def encode_image_data_uri(image_path: Path, *, max_side: int = 2600) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise Pdf2MdError("Pillow is required for direct page OCR fallback") from exc
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def direct_page_ocr(
    image_path: Path,
    *,
    api_port: int,
    model: str,
    timeout_seconds: int | None,
    prompt: str = "Text Recognition:",
) -> str:
    data_uri = encode_image_data_uri(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 0.00001,
        "repetition_penalty": 1.1,
    }
    encoded = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"http://127.0.0.1:{api_port}/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds or 120) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise Pdf2MdError(f"Direct page OCR request failed for {image_path.name}: {exc}") from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise Pdf2MdError(f"Direct page OCR returned an invalid response for {image_path.name}: {str(data)[:500]}") from exc
    return normalize_ocr_markdown(str(content or ""))


def macos_vision_script_path() -> Path:
    return Path(__file__).resolve().parent / "scripts" / "vision_ocr.swift"


def swift_executable() -> str:
    configured = os.environ.get("PDF2MD_SWIFT")
    if configured:
        return configured
    found = shutil.which("swift")
    if found:
        return found
    default = Path("/usr/bin/swift")
    if default.exists():
        return str(default)
    raise Pdf2MdError(
        "macOS Vision OCR fallback requires Swift. Install Xcode Command Line Tools with `xcode-select --install`."
    )


def parse_vision_boxes(raw: str) -> list[VisionTextBox]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Vision OCR JSON root is not a list")
    boxes: list[VisionTextBox] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        boxes.append(
            VisionTextBox(
                text=text,
                min_x=float(item.get("min_x", 0.0)),
                min_y=float(item.get("min_y", 0.0)),
                max_x=float(item.get("max_x", 0.0)),
                max_y=float(item.get("max_y", 0.0)),
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return boxes


def clean_vision_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    watermark_patterns = [
        r"想要.{0,10}(?:海量|投研|投砑).{0,20}",
        r"公众号[:：].*",
        r"微信[:：]\s*Macro[_ ]?Guru",
        r"Macro[_ ]?Guru",
        r"\bcro\s+Guru\b",
        r"[奥奧]\s*KA\s*姆?剃刀",
        r"独家的一手信息[，,]\s*独立的行业思考",
        r"福喫获！?",
    ]
    for pattern in watermark_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"([，,。；;])\s*[與与奥奧]\s*KA\s*$", r"\1", text)
    text = re.sub(r"\s*[與与奥奧]\s*KA\s*$", "", text).strip()
    text = re.sub(r"([。；;])\s*的行业\s*$", r"\1", text)
    return text


def vision_box_is_low_contrast_watermark(gray_image: Any, box: VisionTextBox) -> bool:
    width, height = gray_image.size
    x0 = max(0, int(box.min_x * width) - 2)
    x1 = min(width, int(box.max_x * width) + 2)
    y0 = max(0, int((1.0 - box.max_y) * height) - 2)
    y1 = min(height, int((1.0 - box.min_y) * height) + 2)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = gray_image.crop((x0, y0, x1, y1))
    pixel_source = getattr(crop, "get_flattened_data", crop.getdata)
    pixels = sorted(pixel_source())
    if not pixels:
        return False
    p05 = pixels[min(len(pixels) - 1, int(len(pixels) * 0.05))]
    dark_fraction = sum(value < 180 for value in pixels) / len(pixels)
    return p05 >= 210 and dark_fraction < 0.004


def vision_box_is_metadata(text: str, box: VisionTextBox) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if re.fullmatch(r"\d+", compact) and (box.max_y < 0.08 or box.min_y > 0.90):
        return True
    if compact in {"日", "ex)", "ex）"} and (box.max_y > 0.82 or box.min_y > 0.60):
        return True
    if re.fullmatch(r"J\.?P\.?Morgan", compact, flags=re.IGNORECASE):
        return True
    if compact.startswith("www.") or "jpmorganmarkets.com" in compact.lower():
        return True
    if "分析师认证" in compact and "披露" in compact:
        return True
    if "股票研究" in compact and re.search(r"20\d{2}年", compact):
        return True
    if "@" in text:
        return True
    phone_like = bool(re.search(r"[（(]\d{2}-\d+[）)]|\d{2,4}[- ]\d{3,}", text))
    if phone_like and (box.min_x > 0.58 or box.min_y > 0.86):
        return True
    if box.min_x > 0.58 and re.fullmatch(r"\d{3,5}", compact):
        return True
    if box.min_x > 0.62 and any(
        marker in text
        for marker in (
            "技术-半导体",
            "证券",
            "有限公司",
            "首尔分行",
            "Securities",
            "Ltd",
            "Ac",
        )
    ):
        return True
    if box.min_y > 0.88 and re.search(r"[（(]\d{2}-\d+[）)]|\d{2,4}[- ]\d{3,}", text):
        return True
    return False


def is_vision_heading(line: str) -> bool:
    if len(line) > 36:
        return False
    if line.startswith(("•", "-", "图", "表", "来源")):
        return False
    return not bool(re.search(r"[，。！？；：,.!?;:]", line))


def join_vision_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if re.search(r"[\u4e00-\u9fff）)]$", left) and re.search(r"^[\u4e00-\u9fff（(]", right):
        return left + right
    return left + " " + right


def merge_vision_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            paragraphs.append(current.strip())
        current = ""

    for line in lines:
        line = line.strip()
        if not line:
            flush()
            continue
        boundary = line.startswith(("•", "图", "表", "来源")) or (not current and is_vision_heading(line))
        if boundary:
            flush()
            current = line
            if is_vision_heading(line) or line.startswith(("图", "表", "来源")):
                flush()
            continue
        current = join_vision_text(current, line)
    flush()
    return "\n\n".join(paragraphs)


def format_vision_observations(image_path: Path, boxes: list[VisionTextBox]) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise Pdf2MdError("Pillow is required for macOS Vision OCR cleanup") from exc

    with Image.open(image_path) as image:
        gray = image.convert("L")

    lines: list[str] = []
    for box in sorted(boxes, key=lambda item: (-item.mid_y, item.min_x)):
        text = clean_vision_text(box.text)
        if not text:
            continue
        if vision_box_is_metadata(text, box):
            continue
        if vision_box_is_low_contrast_watermark(gray, box):
            continue
        lines.append(text)
    return normalize_ocr_markdown(merge_vision_lines(lines))


def macos_vision_ocr(
    image_path: Path,
    *,
    timeout_seconds: int | None,
    script_path: Path | None = None,
) -> str:
    script = script_path or macos_vision_script_path()
    if not script.exists():
        raise Pdf2MdError(f"macOS Vision OCR script not found: {script}")

    module_cache = Path(__file__).resolve().parent / ".cache" / "clang"
    module_cache.mkdir(parents=True, exist_ok=True)
    command = [
        swift_executable(),
        "-module-cache-path",
        str(module_cache),
        str(script),
        str(image_path),
        "--json",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionTimeout(f"macOS Vision OCR timed out while processing {image_path.name}") from exc
    except OSError as exc:
        raise Pdf2MdError(f"macOS Vision OCR failed to start for {image_path.name}: {exc}") from exc

    if completed.returncode != 0:
        raise Pdf2MdError(
            "macOS Vision OCR failed for "
            f"{image_path.name} with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    try:
        boxes = parse_vision_boxes(completed.stdout)
    except (ValueError, TypeError, json.JSONDecodeError):
        return normalize_ocr_markdown(completed.stdout)
    return format_vision_observations(image_path, boxes)


def build_glmocr_command(
    image_path: Path,
    config_path: Path | None,
    output_dir: Path,
    *,
    api_port: int,
    model: str,
) -> list[str]:
    command = [
        glmocr_executable(),
        "parse",
        str(image_path),
        "--output",
        str(output_dir),
        "--mode",
        "selfhosted",
        "--layout-device",
        "cpu",
        "--no-layout-vis",
        "--set",
        "pipeline.ocr_api.api_host",
        "127.0.0.1",
        "--set",
        "pipeline.ocr_api.api_port",
        str(api_port),
        "--set",
        "pipeline.ocr_api.model",
        model,
        "--set",
        "pipeline.ocr_api.api_path",
        "/chat/completions",
        "--set",
        "pipeline.ocr_api.verify_ssl",
        "false",
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    return command


def run_glmocr_cli(
    image_path: Path,
    config_path: Path | None,
    output_dir: Path,
    *,
    timeout_seconds: int | None,
    api_port: int,
    model: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_glmocr_command(image_path, config_path, output_dir, api_port=api_port, model=model)
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionTimeout(f"GLM-OCR timed out while processing {image_path.name}") from exc
    if completed.returncode != 0:
        raise Pdf2MdError(
            "GLM-OCR failed for "
            f"{image_path.name} with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    markdown_path = newest_markdown_file(output_dir)
    if markdown_path is None:
        print(
            f"warning: GLM-OCR produced no Markdown output for {image_path.name}; trying fallback OCR",
            file=sys.stderr,
        )
        return ""
    return normalize_ocr_markdown(markdown_path.read_text(encoding="utf-8"))


OcrRunner = Callable[[Path, Path | None, Path, int | None], str]


def build_output_markdown(
    job: PdfJob,
    *,
    page_markdowns: list[str],
    page_count: int,
    model: str,
    watermark: WatermarkSummary,
    timeout_seconds: int | None,
    fallback_engines: tuple[str, ...] = (),
) -> str:
    fields = {
        "schema_version": "1.0",
        "status": "completed",
        "source_path": str(job.source_path),
        "target_path": str(job.target_path),
        "relative_path": str(job.relative_path),
        "processed_at": now_local(),
        "model": model,
        "page_count": page_count,
        "watermark_preprocess": watermark.mode,
        "watermark_applied_pages": watermark.applied_pages,
        "watermark_fallback_pages": watermark.fallback_pages,
        "timeout_seconds": timeout_seconds,
        "fallback_ocr_engines": ", ".join(fallback_engines) if fallback_engines else "none",
    }
    frontmatter = "\n".join(f"{key}: {yaml_scalar(value)}" for key, value in fields.items())
    title = job.source_path.stem
    body = "\n\n".join(
        f"<!-- page {index} -->\n\n{markdown.strip()}"
        for index, markdown in enumerate(page_markdowns, start=1)
        if markdown.strip()
    ).strip()
    if not body:
        body = "_No OCR content returned._"
    return f"---\n{frontmatter}\n---\n\n# {title}\n\n{body}\n"


def convert_pdf_job(
    job: PdfJob,
    *,
    work_root: Path,
    config_path: Path | None = None,
    model: str = DEFAULT_MODEL,
    api_port: int = 8080,
    preprocess_watermark: bool = True,
    timeout_seconds: int | None = DEFAULT_TIMEOUT_SECONDS,
    deadline: float | None = None,
    ocr_runner: OcrRunner | None = None,
) -> ConversionResult:
    ocr_runner = ocr_runner or (
        lambda image, config, output, timeout: run_glmocr_cli(
            image,
            config,
            output,
            timeout_seconds=timeout,
            api_port=api_port,
            model=model,
        )
    )
    work_dir = work_root / job.job_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    page_images, watermark = prepare_page_images(job.source_path, work_dir, preprocess_watermark=preprocess_watermark)
    page_markdowns: list[str] = []
    fallback_engines: list[str] = []
    for index, image in enumerate(page_images, start=1):
        timeout_for_page = remaining_seconds(deadline)
        page_output_dir = work_dir / "ocr" / f"page-{index:04d}"
        markdown = normalize_ocr_markdown(ocr_runner(image, config_path, page_output_dir, timeout_for_page))
        fallback_errors: list[str] = []
        if not markdown_has_meaningful_content(markdown):
            try:
                fallback_timeout = remaining_seconds(deadline)
                markdown = direct_page_ocr(
                    image,
                    api_port=api_port,
                    model=model,
                    timeout_seconds=fallback_timeout,
                )
                if markdown_has_meaningful_content(markdown):
                    fallback_engines.append("direct_mlx_vlm")
                else:
                    fallback_errors.append("direct_mlx_vlm returned empty content")
            except ConversionTimeout:
                raise
            except Exception as exc:
                fallback_errors.append(f"direct_mlx_vlm failed: {exc}")
                markdown = ""
        if not markdown_has_meaningful_content(markdown):
            try:
                fallback_timeout = remaining_seconds(deadline)
                markdown = macos_vision_ocr(image, timeout_seconds=fallback_timeout)
                if markdown_has_meaningful_content(markdown):
                    fallback_engines.append("macos_vision")
                else:
                    fallback_errors.append("macos_vision returned empty content")
            except ConversionTimeout:
                raise
            except Exception as exc:
                fallback_errors.append(f"macos_vision failed: {exc}")
                markdown = ""
        if not markdown_has_meaningful_content(markdown):
            details = "; ".join(fallback_errors)
            suffix = f" ({details})" if details else ""
            raise Pdf2MdError(f"OCR returned no meaningful content for {job.relative_path} page {index}{suffix}")
        page_markdowns.append(markdown)

    markdown = build_output_markdown(
        job,
        page_markdowns=page_markdowns,
        page_count=len(page_images),
        model=model,
        watermark=watermark,
        timeout_seconds=timeout_seconds,
        fallback_engines=tuple(dict.fromkeys(fallback_engines)),
    )
    atomic_write_text(job.target_path, markdown)
    return ConversionResult(
        output_path=job.target_path,
        page_count=len(page_images),
        watermark=watermark,
        fallback_engines=tuple(dict.fromkeys(fallback_engines)),
    )
