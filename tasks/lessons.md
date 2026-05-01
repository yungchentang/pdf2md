# Lessons

## 2026-05-01 - Initial repo push must include runnable code

- When the user asks to push a new project repo for others to clone and use, do not mechanically follow a README-only bootstrap snippet if the local workspace already contains implementation files.
- Before pushing, inspect untracked files and identify the minimal runnable project set: source code, helper scripts, dependency manifests, tests, and `.gitignore`.
- In the final verification, confirm the remote commit contains the runnable code path, not only documentation.

## 2026-04-28 - Match the requested conversion engine

- When the user asks to use a specific skill or model path for PDF-to-Markdown conversion, do not silently substitute the repo's existing local OCR pipeline just because it is available.
- Before converting user-provided PDFs, state the actual engine path up front: local GLM-OCR/MLX, macOS Vision fallback, or GPT model reading rendered pages.
- If the requested output is "GPT reads and converts", render PDF pages and route page images/text through the GPT path, then label the generated Markdown accordingly.

## 2026-04-28 - venv executable discovery

- Do not use `Path(sys.executable).resolve()` to locate sibling console scripts inside a venv. Homebrew Python venvs can symlink `python` to `/opt/homebrew/...`, so `resolve()` escapes the venv and misses scripts like `.venv-sdk/bin/glmocr`.
- Prefer `Path(sys.prefix) / "bin" / <tool>` for venv-owned console scripts, then fall back to `Path(sys.executable).parent`.
- When a subprocess server exits during startup, include recent stdout/stderr tails in the user-facing error. A generic `exited early with code 1` hides root causes like `No Metal device available`.

## 2026-04-28 - GLM-OCR self-hosted config

- Do not replace GLM-OCR's packaged `config.yaml` with a minimal custom config. The packaged config carries required self-hosted layout defaults such as `pipeline.layout.model_dir`, `label_task_mapping`, and `id2label`.
- For MLX self-hosted use, call `glmocr parse --mode selfhosted` and override only OCR API fields with `--set pipeline.ocr_api...`; this preserves the official PP-DocLayoutV3 layout configuration.
- Add config-level smoke checks for external SDK integrations before running the expensive model path.

## 2026-04-28 - Empty OCR output must fail or fallback

- Never mark an OCR output as completed just because the subprocess returned exit code 0. Inspect the Markdown body and reject empty code fences, image-only placeholders, and wrapper-only output.
- GLM-OCR's region pipeline can return empty Markdown content while still producing JSON layout regions and HTTP 200 responses from the VLM server. Treat that as an OCR failure for the page.
- Add a whole-page direct OCR fallback when layout-region OCR is empty, then fail visibly if the fallback is also empty.

## 2026-04-28 - Use a local system OCR fallback for MLX empty output

- When the MLX GLM-OCR path returns HTTP 200 with empty content on a legible page image, do not keep tuning the same failing request in place. Add an independent local OCR fallback so the worker still produces useful output.
- On Apple Silicon/macOS deployments, macOS Vision OCR is a practical final fallback for text-heavy protected PDFs because it uses the local Vision framework and does not require a cloud API.
- Preserve the GLM-OCR path as the primary engine and record fallback engines in Markdown frontmatter so lower-fidelity fallback output is observable.

## 2026-04-28 - Vision OCR needs layout-aware cleanup

- Do not treat Vision OCR plain-text ordering as final Markdown. Whole-page OCR can merge right sidebars, headers, footers, and low-contrast watermark/ad text into the article body.
- Ask Vision for bounding boxes and filter artifacts by position, contrast, and metadata patterns before building Markdown.
- Merge wrapped PDF body lines after filtering; otherwise even correct OCR feels choppy because every rendered line break becomes a reading break.
