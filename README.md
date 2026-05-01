# pdf2md

Local Apple Silicon worker for converting PDF files to Markdown with GLM-OCR.

This project follows the same operational shape as `podcast2text`: a small local
worker, explicit setup, one-shot runs, observable status, and optional macOS
LaunchAgent scheduling.

## What Runs Locally

`pdf2md` uses GLM-OCR through the Apple Silicon MLX path:

- `.venv-mlx` runs `mlx-vlm` and loads `mlx-community/GLM-OCR-bf16`.
- `.venv-sdk` runs this CLI plus the `glmocr` SDK.
- Each `run` starts a local `mlx-vlm` server only for that run.
- The server is stopped when the run finishes, fails, is interrupted, or times out.
- If GLM-OCR returns an empty page, the worker tries direct whole-page GLM-OCR,
  then macOS Vision OCR as a local fallback. The fallback uses Apple's built-in
  Vision framework through Swift and does not call a cloud service.

The server is bound to `127.0.0.1` and is not intended to be exposed publicly.

## Setup

```bash
cd /Users/kumi/Projects/pdf2md
python3 pdf2md.py setup
```

The first setup can take a while because it installs dependencies and warms up
the GLM-OCR model. To install dependencies without warming the model:

```bash
python3 pdf2md.py setup --skip-model-warmup
```

## Folder Layout

The source folder can contain PDFs directly:

```text
source/
  report.pdf
```

or date folders:

```text
source/
  2026-04-28/
    report.pdf
```

The target folder mirrors that layout:

```text
target/
  report.md
  2026-04-28/
    report.md
```

## First Full Backfill

Run the first full conversion manually:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --all \
  --timeout-seconds 21600
```

`--timeout-seconds` is a per-file budget. If one PDF times out, that file is
marked failed and the next PDF starts with a fresh timer, so overnight backfills
can continue through a bad or unusually slow file. Use `--no-timeout` only for a
manual full backfill when you are watching the machine:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --all \
  --no-timeout
```

## Recurring Runs

Recurring automation should scan only recent date folders and always use a
per-file timeout:

```bash
.venv-sdk/bin/python pdf2md.py install-schedule \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --lookback-days 3 \
  --timeout-seconds 3000 \
  --load \
  --start-now
```

This writes:

```text
~/Library/LaunchAgents/com.kumi.pdf2md.plist
```

and runs hourly by default. Scheduled `--all` backfills are blocked unless
`--allow-scheduled-backfill` is explicitly passed, and scheduled `--no-timeout`
is never allowed.

Preview the LaunchAgent without writing it:

```bash
.venv-sdk/bin/python pdf2md.py install-schedule \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --dry-run
```

## Normal Run

Scan the latest three date folders:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder
```

Scan one date:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --date 2026-04-28
```

Process two PDFs concurrently:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --all \
  --workers 2
```

`--workers` controls concurrent PDF jobs. The default is `1`. Start with
`--workers 2` on Apple Silicon; higher values can increase memory pressure and
may not help if the local MLX server is already the bottleneck.

Dry-run the current scan without starting OCR:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --all \
  --dry-run
```

By default, outputs contain only converted page content. Use `--header` only if
you want YAML frontmatter and a generated `# filename` title for debugging:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /path/to/pdf-folder \
  --target /path/to/md-folder \
  --all \
  --header
```

Completed Markdown files are skipped when they have meaningful content. Headered
outputs are also skipped when their frontmatter contains:

```yaml
status: completed
```

Use `--force` to regenerate existing completed outputs. This is required if you
want to remove headers from Markdown files that were already generated earlier.

If the GLM-OCR SDK region pipeline returns empty Markdown for a page, the worker
falls back to direct whole-page OCR against the same local `mlx-vlm` server. If
that also returns empty content, it falls back to macOS Vision OCR. With
`--header`, the fallback is recorded in frontmatter:

```yaml
fallback_ocr_engines: "macos_vision"
```

If macOS cannot find `swift`, install Xcode Command Line Tools:

```bash
xcode-select --install
```

## Watermark Handling

Watermark preprocessing is enabled by default. It conservatively lightens
low-contrast gray watermark-like pixels while preserving dark text and table
lines. Disable it with:

```bash
--no-preprocess-watermark
```

If preprocessing fails or looks too destructive, the page falls back to the
original image and the Markdown frontmatter records the fallback.

## Status and Logs

```bash
.venv-sdk/bin/python pdf2md.py status
.venv-sdk/bin/python pdf2md.py status --logs
```

Runtime state is written to:

```text
.cache/status.json
```

Logs are written to:

```text
logs/launchd.out.log
logs/launchd.err.log
logs/mlx-vlm.out.log
logs/mlx-vlm.err.log
```

## Example with This Repo

The current sample PDF lives under `pdfs/`. Because it is not inside a date
folder, use `--all`:

```bash
.venv-sdk/bin/python pdf2md.py run \
  --source /Users/kumi/Projects/pdf2md/pdfs \
  --target /Users/kumi/Projects/pdf2md/outputs \
  --all
```

## References

- GLM-OCR: https://github.com/zai-org/GLM-OCR
- GLM-OCR MLX deployment guide: https://github.com/zai-org/GLM-OCR/blob/main/examples/mlx-deploy/README.md
