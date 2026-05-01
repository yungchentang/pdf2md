# pdf2md local OCR worker

## 2026-05-01 - Single total progress bar

### Plan
- [x] Replace per-file progress bars with one total completed-files bar.
- [x] Keep current worker/page text as activity status, not separate bars.
- [x] Update heartbeat to show the same total bar plus active job details.
- [x] Update tests for the new output shape.
- [x] Run compile and unit tests.
- [x] Document review results here.

### Review
- Removed per-file progress bars from job start messages.
- New files now print `start <n>/<total>: <file>` as activity text.
- The only progress bar now represents aggregate completed PDFs: `done=<n>, processed=<n>, failed=<n>`.
- Heartbeats reuse the aggregate progress bar and append active worker/page details.
- Verification passed: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`; `.venv-sdk/bin/python -m unittest discover -s tests` ran 37 tests OK.

## 2026-05-01 - Reduce progress noise and show model startup

### Plan
- [x] Remove duplicate normal-run `pending ...` output.
- [x] Submit only active worker jobs instead of printing the entire queue as processing.
- [x] Add visible model/server startup status and heartbeat while `mlx-vlm` loads.
- [x] Keep dry-run useful by listing pending files there.
- [x] Run compile and unit tests.
- [x] Document review results here.

### Review
- Normal runs now print one scan summary instead of every `pending ...` and `skip ...` line.
- Dry-runs still list pending/skipped files because there is no processing progress output in that mode.
- Worker scheduling now starts only up to `--workers` active jobs. It no longer prints the whole queue as `processing`.
- Server startup prints `starting mlx-vlm server ...`, `server: still starting ...`, and `server: mlx-vlm server ready ...` messages while the model loads.
- Verification passed: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`; `.venv-sdk/bin/python -m unittest discover -s tests` ran 36 tests OK.

## 2026-05-01 - Live progress for workers

### Plan
- [x] Confirm why multi-file progress appears stuck.
- [x] Add page-level progress callbacks from conversion workers.
- [x] Let the main run loop print worker progress events while jobs are still running.
- [x] Add heartbeat output while long OCR calls are active.
- [x] Preserve final processed/failed accounting and status JSON.
- [x] Add focused tests for progress callback behavior.
- [x] Run compile and unit tests.
- [x] Document review results here.

### Review
- Root cause: the worker implementation submitted all PDF futures, then blocked in `as_completed()`. Progress only moved when a whole PDF completed or failed.
- Added `progress_callback` to `convert_pdf_job()` so workers emit `pages-ready`, `page-start`, and `page-done` events.
- The main run loop now checks worker events every second and prints page-level progress while PDFs are still running.
- Added a 30-second heartbeat for long OCR calls so the terminal shows active jobs even when a single page is taking a long time.
- Status JSON now includes `active_pages` during processing.
- Verification passed: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`; `.venv-sdk/bin/python -m unittest discover -s tests` ran 35 tests OK.

## 2026-05-01 - Workers option for parallel PDFs

### Plan
- [x] Confirm `--workers` is not currently supported.
- [x] Add `--workers` CLI validation with default `1`.
- [x] Process multiple pending PDFs concurrently when `--workers > 1`, while keeping one shared MLX server.
- [x] Keep per-file timeout, failure collection, no-header, and progress output working under parallel execution.
- [x] Pass `--workers` through scheduled LaunchAgent generation.
- [x] Add focused tests for parsing, validation, scheduling, and parallel dispatch.
- [x] Run compile and unit tests.
- [x] Document review results here.

### Review
- `--workers` did not exist before this change.
- Added `--workers` to run/scheduled arguments with default `1` and validation that it must be positive.
- `run_once()` now uses a `ThreadPoolExecutor` for pending PDF jobs. Each job keeps its own per-file timeout deadline and writes to its own work directory.
- The implementation intentionally keeps one shared `mlx-vlm` server on the configured port to avoid loading multiple model copies by default.
- Final status JSON records `workers`, and in-progress status records completed/active counts.
- LaunchAgent generation passes `--workers <n>` through.
- README documents `--workers 2` and warns that higher values can increase memory pressure.
- Verification passed: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`; `.venv-sdk/bin/python -m unittest discover -s tests` ran 33 tests OK.

## 2026-05-01 - Optional Markdown header

### Plan
- [x] Confirm where output Markdown header is generated.
- [x] Add a CLI option to disable frontmatter and title header for converted files.
- [x] Preserve skip behavior for no-header outputs so completed files are not reprocessed every run.
- [x] Pass the option through scheduled runs.
- [x] Add focused tests and update README.
- [x] Run compile and unit tests.
- [x] Document review results here.

### Review
- The generated header came from `build_output_markdown()`: YAML frontmatter plus a `# <filename>` title.
- Added `--no-header` / `--header` to shared run arguments. Default behavior stays unchanged.
- `--no-header` writes only page markers and OCR body content, without frontmatter or title.
- Updated skip logic so no-header outputs with meaningful Markdown content are skipped on later no-header runs.
- Scheduled runs can pass through `--no-header`; LaunchAgent generation includes it when selected.
- Updated README with the no-header command example.
- Verification passed: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`; `.venv-sdk/bin/python -m unittest discover -s tests` ran 30 tests OK.

## 2026-05-01 - Per-file timeout and progress

### Plan
- [x] Confirm current timeout behavior and progress/status output.
- [x] Change `run --timeout-seconds` from a whole-run budget to a per-file budget so each new PDF gets a fresh timer.
- [x] Keep single-file timeout as a failed file and continue the batch instead of aborting the run.
- [x] Add visible per-file progress output while preserving existing status-file fields.
- [x] Add focused tests for per-file timeout continuation and progress formatting.
- [x] Run compile and unit tests.
- [x] Document review results here.

### Review
- Root cause: `run_once()` created one run-level deadline before processing and passed that same deadline into every PDF conversion. A `ConversionTimeout` also escaped the per-file loop and ended the whole batch with exit code 124.
- Changed `run --timeout-seconds` to mean per-file timeout: each pending PDF now gets a fresh `file_deadline`, and a timeout records that PDF as failed before continuing to the next file.
- Added console progress lines such as `progress [######------------------] 1/4 ( 25%) processed file.md` for processing, processed, and failed states.
- Kept the existing status-file flow and final `done-with-failures` behavior for batches where some files fail.
- Updated README to document per-file timeout semantics for manual backfills and scheduled runs.
- Verification passed: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`; `.venv-sdk/bin/python -m unittest discover -s tests` ran 27 tests OK.

## 2026-05-01 - Correction: push runnable code

### Plan
- [x] Acknowledge the initial push only contained `README.md` and is not enough for a clone-and-run workflow.
- [x] Stage runnable project files: `.gitignore`, Python code, requirements, scripts, tests, and task/lesson records.
- [x] Keep generated/local artifacts out of git: venvs, cache, logs, outputs, PDFs, and Python bytecode.
- [x] Run verification before committing.
- [x] Commit and push the code to `origin/main`.
- [x] Verify GitHub tracking state and latest commit.

### Review
- Pushed commit `47f6770 Add runnable pdf2md worker` to `origin/main`.
- Remote now contains the runnable project files: `.gitignore`, `pdf2md.py`, `pdf_to_markdown.py`, `requirements-mlx.txt`, `requirements-sdk.txt`, `scripts/vision_ocr.swift`, and `tests/test_pdf2md.py`.
- Included `tasks/lessons.md` and `tasks/todo.md` so the correction and verification trail are preserved.
- Verification passed before commit: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`; `.venv-sdk/bin/python -m unittest discover -s tests` ran 25 tests OK.
- Verified latest state after push: `47f6770 (HEAD -> main, origin/main) Add runnable pdf2md worker`.
- Generated/local artifacts remain out of git via `.gitignore`: `.cache/`, `.venv*/`, `logs/`, `outputs/`, `codex_output/`, `pdfs/`, `__pycache__/`, and related test/cache outputs.

## 2026-05-01 - Initial GitHub push

### Plan
- [x] Confirm `/Users/kumi/Projects/pdf2md` is not already a git repository.
- [x] Confirm `README.md` already starts with `# pdf2md` and avoid appending a duplicate heading.
- [x] Initialize git repository and stage `README.md`.
- [x] Create initial commit with message `first commit`.
- [x] Set default branch to `main`.
- [x] Add GitHub remote `git@github.com:yungchentang/pdf2md.git`.
- [x] Push `main` to `origin` with upstream tracking.
- [x] Verify local branch, remote, and push result.

### Review
- Created root commit `67c9af4 first commit` with `README.md` only.
- Did not append another `# pdf2md` line because `README.md` already started with that heading.
- Added `origin` as `git@github.com:yungchentang/pdf2md.git`.
- Pushed `main` to GitHub and set upstream tracking: `main...origin/main`.
- Verified latest commit is `67c9af4 (HEAD -> main, origin/main) first commit`.
- Left existing untracked project files untouched: `.gitignore`, source files, outputs, scripts, tasks, and tests.

## 2026-04-28 - Redo codex_output with GPT vision conversion

### Plan
- [x] Treat rendered PDF page images as the source for GPT-readable conversion, not the existing local OCR Markdown.
- [x] Overwrite `/Users/kumi/Projects/pdf2md/codex_output` with GPT-converted Markdown.
- [x] Exclude watermark/contact/sidebar/header/footer/page-number artifacts from the Markdown body.
- [x] Verify output file count, page markers, meaningful content, and forbidden watermark/contact patterns.
- [x] Document commands, output paths, and verification results in this file.

### Review
- Overwrote `/Users/kumi/Projects/pdf2md/codex_output/摩根大通_一季度韩国&日本内存厂商业绩前瞻：核心问题梳理+20260422.md` with GPT/vision-converted Markdown.
- The conversion source was the rendered page images under `.cache/work/pdf-2e44ee7c44a2e244/pages-cleaned/`, inspected directly as images; the prior local OCR Markdown was not used as the authoritative conversion source.
- Output frontmatter now records `conversion_engine: "gpt_vision_from_rendered_pages"` and no longer records the prior MLX/macOS Vision OCR engine fields.
- Output count matches input count: 1 PDF under `pdfs/`, 1 Markdown file under `codex_output/`.
- Verified PDF page count is 3 and Markdown page markers are `<!-- page 1 -->`, `<!-- page 2 -->`, and `<!-- page 3 -->`.
- Verified output content size: 57 lines, 3889 chars, 3040 non-whitespace body chars.
- Forbidden artifact scan returned no matches for prior local engine markers, faint overlay terms, Macro/Guru, 微信/公众号, JPM contact emails/sites, known analyst/contact names, phone fragments, and common sidebar/footer artifacts.

## 2026-04-28 - Convert pdfs/ into codex_output/

### Plan
- [x] Inspect all PDFs under `/Users/kumi/Projects/pdf2md/pdfs`.
- [x] Generate Markdown files under `/Users/kumi/Projects/pdf2md/codex_output`, mirroring input filenames.
- [x] Exclude watermark/contact/sidebar/header/footer artifacts from the Markdown body.
- [x] Verify each generated Markdown file has completed frontmatter, meaningful content, expected page count, and no known watermark patterns.
- [x] Document commands, output paths, and verification results in this file.

### Review
- Found 1 source PDF under `/Users/kumi/Projects/pdf2md/pdfs` and generated 1 Markdown file under `/Users/kumi/Projects/pdf2md/codex_output`.
- Output: `/Users/kumi/Projects/pdf2md/codex_output/摩根大通_一季度韩国&日本内存厂商业绩前瞻：核心问题梳理+20260422.md`.
- Fixed a CLI fallback bug where `glmocr parse` could exit successfully without writing a Markdown file and prevent direct/Vision fallback from running.
- Conversion command passed after the fallback fix: `.venv-sdk/bin/python pdf2md.py run --source pdfs --target codex_output --all --force --timeout-seconds 3000`; summary was `discovered=1, skipped=0, processed=1, failed=0`.
- Verified frontmatter: `status: completed`, `page_count: 3`, `watermark_preprocess: "applied"`, `watermark_applied_pages: 3`, `watermark_fallback_pages: 0`, `fallback_ocr_engines: "macos_vision"`.
- Verified structure and content: PDF page count is 3, Markdown page markers are 3, file length is 111 lines / 3693 chars, body has 3013 non-whitespace chars.
- Body-only watermark/contact scan returned zero matches for Macro/Guru, 微信/公众号, watermark/watermark words, promo watermark phrases, JPM contact emails/sites, and known analyst contact/sidebar names.
- Visual inspection of cleaned page PNGs confirmed the source still contains faint watermark/contact/sidebar elements, while the Markdown output did not include those artifacts.
- Verified the per-run MLX server was not left running: `curl -sS --max-time 2 http://127.0.0.1:8080/health` failed to connect.
- Test pass: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests`, focused fallback unittest, and `.venv-sdk/bin/python -m unittest discover -s tests` passed; current unit test count is 25.

## Plan
- [x] Create a deployable Python CLI with `setup`, `run`, `status`, and `install-schedule`.
- [x] Implement source scanning for root PDFs, date folders, `--all`, `--date`, and `--lookback-days`.
- [x] Implement target path mapping and skip logic for completed Markdown outputs.
- [x] Implement per-run `mlx-vlm` server lifecycle with startup health checks and guaranteed teardown.
- [x] Add timeout handling for run and OCR subprocesses.
- [x] Add conservative watermark preprocessing and metadata recording.
- [x] Add LaunchAgent generation with safe scheduling defaults.
- [x] Document setup, manual backfill, recurring automation, and status checks.
- [x] Add unit tests for scanning, skip logic, CLI/schedule safety, server cleanup, and timeout paths.
- [x] Verify with tests and the bundled sample PDF where possible without downloading the model.
- [x] Add a deployable fallback for pages where GLM-OCR/MLX returns empty content despite HTTP 200.
- [x] Verify the sample PDF no longer fails on empty OCR output.
- [x] Document the fallback behavior and capture the correction in lessons.
- [x] Improve Vision fallback readability by filtering watermark/sidebar/header/footer artifacts.
- [x] Regenerate the sample Markdown and verify obvious watermark/contact/sidebar noise is reduced.

## Review
- `python3 -m compileall pdf2md.py pdf_to_markdown.py tests` passed.
- `python3 -m unittest discover -s tests` passed: 18 tests.
- `pdf2md.py run --source pdfs --target /private/tmp/pdf2md-output --all --dry-run` found the bundled root-level sample PDF and wrote finished dry-run status.
- `prepare_page_images(...)` on the bundled sample extracted 3 pages and applied conservative watermark preprocessing to all 3 pages.
- Full GLM-OCR model execution was not run in this pass because `.venv-mlx`, `.venv-sdk`, and model weights are not installed yet; `setup` now owns that path.
- Fixed a venv script discovery bug where `Path(sys.executable).resolve()` escaped `.venv-sdk` through the Homebrew Python symlink and caused `glmocr` to be invoked as a missing PATH command.
- Improved `mlx-vlm` startup failures to include recent server log tails; the Codex sandbox currently reports `No Metal device available`, while the user's normal Terminal may still have Metal access.
- Fixed GLM-OCR self-hosted config handling: the worker now preserves GLM-OCR's packaged layout defaults and uses `--mode selfhosted` plus `--set pipeline.ocr_api...` overrides instead of a minimal YAML that omitted `pipeline.layout.model_dir`.
- Verified config smoke check: `layout_model=PaddlePaddle/PP-DocLayoutV3_safetensors`, `layout_device=cpu`, `ocr_path=/chat/completions`, `ocr_model=mlx-community/GLM-OCR-bf16`.
- Fixed empty-success handling: placeholder Markdown with only empty code fences is no longer treated as completed, so the existing bad output is discovered as pending on the next run.
- Added whole-page direct OCR fallback against the same local `mlx-vlm` server when GLM-OCR SDK region OCR returns no meaningful page content.
- Added macOS Vision OCR as a final local fallback for pages where both GLM-OCR SDK region OCR and direct whole-page GLM-OCR return empty content.
- Verified the fallback on the bundled sample PDF with a forced-empty GLM path: 3 pages processed, watermark preprocessing applied to all pages, and `fallback_ocr_engines: "macos_vision"` recorded.
- Verified the full CLI command against `/Users/kumi/Projects/pdf2md/pdfs` -> `/Users/kumi/Projects/pdf2md/outputs` with `--force`: `discovered=1, skipped=0, processed=1, failed=0`.
- Verified the generated sample Markdown has 134 lines and a completed frontmatter body; a follow-up dry-run skipped it as already completed.
- Verified the per-run MLX server was not left running after the full CLI run: `curl http://127.0.0.1:8080/health` failed to connect.
- Current test pass: `.venv-sdk/bin/python -m compileall pdf2md.py pdf_to_markdown.py tests` and `.venv-sdk/bin/python -m unittest discover -s tests` passed; unit test count is now 23.
- Improved macOS Vision fallback to consume OCR bounding boxes as JSON, filter low-contrast watermark boxes, drop contact/sidebar/header/footer metadata, and merge wrapped PDF lines into smoother paragraphs.
- Regenerated the sample Markdown after cleanup; obvious watermark/contact/sidebar terms (`公众号`, `微信`, `Macro`, `Guru`, `jpmorgan.com`, `技术-半导体`, `758`, `6157`, `Ac`) are no longer present except normal source text `来源`.
- Current sample Markdown is 111 lines after paragraph merging and remains `status: completed`; dry-run skips it as already completed.
- Current test pass after cleanup changes: `.venv-sdk/bin/python -m compileall pdf_to_markdown.py tests` and `.venv-sdk/bin/python -m unittest discover -s tests` passed; unit test count is now 24.
