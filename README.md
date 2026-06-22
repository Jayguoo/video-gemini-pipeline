# Video Understanding Pipeline (Gemini)

Extract structured understanding from coding/tutorial videos using Gemini + OCR + Whisper transcription.

## Features

- **Frame extraction** — OpenCV or ffmpeg with scene-aware or interval sampling
- **OCR** — Tesseract on keyframes to capture on-screen text (commands, errors, paths)
- **Transcription** — Whisper local transcription or pre-supplied transcript
- **Coding evidence extraction** — Regex scan for commands, file paths, errors, URLs, package names
- **Gemini analysis** — Uploads video and optional keyframes, with configurable media resolution, chunked processing for long videos, and structured JSON output
- **Verification pass** — Optional second Gemini call to catch missed evidence

## Requirements

- Python 3.10+
- Google Gemini API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY` env var)
- Optional: ffmpeg, Tesseract OCR, Whisper (detected automatically)

## Install

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Usage

```bash
# Basic analysis
python video_gemini_pipeline.py video.mp4

# Coding video mode (structured JSON output)
python video_gemini_pipeline.py video.mp4 --coding-mode

# Deep crunch — high-detail coding video analysis
python video_gemini_pipeline.py video.mp4 --deep-crunch

# Long video: process in 5-min chunks then merge
python video_gemini_pipeline.py video.mp4 --chunk-minutes 5

# All options
python video_gemini_pipeline.py video.mp4 \
  --coding-mode \
  --high-res \
  --upload-keyframes \
  --run-whisper \
  --structured-json \
  --verify-pass \
  --chunk-minutes 3
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `gemini-3.5-flash` | Gemini model |
| `--coding-mode` | off | Coding-video prompt/JSON schema |
| `--deep-crunch` | off | High-detail preset |
| `--high-res` | off | High media resolution |
| `--upload-keyframes` | off | Upload keyframes to Gemini |
| `--max-frames` | 24 | Max extracted frames |
| `--frame-interval` | 10s | Seconds between sampled frames |
| `--run-whisper` | off | Local Whisper transcription |
| `--chunk-minutes` | 0 | Chunk long videos (N minutes) |
| `--structured-json` | off | Schema-constrained JSON output |
| `--verify-pass` | off | Second verification pass |
