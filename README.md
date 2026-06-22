# Video Understanding Pipeline (Gemini)

Extract structured, timestamped understanding from coding and tutorial videos using Gemini + OCR + Whisper transcription.

## Why This Pipeline

Most approaches to video analysis fall into two camps — upload raw video and hope the model sees everything, or extract text and lose visual context. This pipeline combines both:

- **Preprocessing enriches the model.** OCR captures terminal output, error messages, and code on screen. Whisper transcribes speech. Regex evidence extraction surfaces commands, paths, and errors before they reach the model. Gemini gets this context alongside the video, so it doesn't have to squint at small text or miss a spoken command.
- **Chunk-then-merge.** Long videos are split into overlapping segments, analyzed independently, then merged — no truncation, no lost context at the end of a 30-minute coding session.
- **Verification pass.** A second Gemini call cross-checks the analysis against the raw OCR and transcript, catching hallucinations and missed evidence.
- **Structured output.** Optional JSON schema enforces consistent fields (timeline, commands, files, errors) for downstream tooling.

## Why Gemini

Other models accept video, but Gemini's native video understanding has advantages that this pipeline specifically exploits:

- **File API with temporal grounding.** Upload a video once; Gemini processes it server-side and can reference any timestamp without frame-by-frame client-side sampling. The pipeline supplies `VideoMetadata` (FPS, start/end offsets) so Gemini knows exactly which time window it's analyzing.
- **Media resolution control.** For coding videos with small terminal fonts, `--high-res` forces `MEDIA_RESOLUTION_HIGH` or `ULTRA_HIGH` — a setting most multimodal models don't expose.
- **Schema-constrained JSON.** `response_schema` makes Gemini return perfectly valid JSON matching the pipeline's schema on the first attempt, no parsing needed.
- **Code execution tool.** `--code-execution` lets Gemini run code inside the analysis to verify commands or reproduce errors.
- **Chunked content.** Gemini's long context window handles the video plus uploaded keyframes plus full transcript in a single request. When combined with the chunk-then-merge strategy, there's no practical video length limit.

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
