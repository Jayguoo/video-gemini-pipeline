import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

CODING_TASK = """Analyze this coding video as a debugging/tutorial assistant.

Prioritize:
- exact commands typed
- filenames, paths, tabs, and package names
- code changes and pasted snippets
- terminal errors, stack traces, and warnings
- versions, config keys, environment variables, and URLs
- UI clicks only when they affect the coding task

Return:
1. Goal of the video.
2. Step-by-step timestamped timeline.
3. Commands/code shown.
4. Errors and fixes.
5. Final working state.
6. Anything unclear or unreadable."""

CODING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "goal": {"type": "string"},
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "event": {"type": "string"},
                    "visual_evidence": {"type": "string"},
                    "speech_evidence": {"type": "string"},
                    "ocr_evidence": {"type": "string"},
                    "confidence": {"type": "string"},
                },
                "required": ["timestamp", "event", "confidence"],
            },
        },
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "command": {"type": "string"},
                    "purpose": {"type": "string"},
                },
                "required": ["command"],
            },
        },
        "files_changed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "path": {"type": "string"},
                    "change": {"type": "string"},
                },
                "required": ["path", "change"],
            },
        },
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "error": {"type": "string"},
                    "likely_cause": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["error"],
            },
        },
        "final_state": {"type": "string"},
        "unclear": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "timeline", "commands", "files_changed", "errors", "unclear"],
}

GENERIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "event": {"type": "string"},
                    "visual_evidence": {"type": "string"},
                    "speech_evidence": {"type": "string"},
                    "ocr_evidence": {"type": "string"},
                    "confidence": {"type": "string"},
                },
                "required": ["timestamp", "event", "confidence"],
            },
        },
        "key_details": {"type": "array", "items": {"type": "string"}},
        "unclear": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "timeline", "key_details", "unclear"],
}

COMMAND_RE = re.compile(
    r"(?i)(?:^|\s)(?:\$|>|PS [^>]+>|C:\\[^>]+>|"
    r"python|py|pip|uv|node|npm|npx|pnpm|yarn|bun|git|gh|docker|docker-compose|"
    r"kubectl|terraform|cargo|rustc|go|java|javac|mvn|gradle|dotnet|code|codex|agy)"
    r"(?:\s+[^\n\r]{1,240})?"
)
PATH_RE = re.compile(
    r"(?i)(?:[A-Z]:\\[^\s\"'<>|]+|(?:\.{0,2}/)?[\w.\-]+(?:/[\w.\-]+)+|"
    r"\b[\w.\-]+\.(?:py|js|jsx|ts|tsx|json|toml|yaml|yml|md|txt|html|css|scss|rs|go|java|cs|cpp|c|h|hpp|sh|ps1|bat|cmd|sql|env)\b)"
)
ERROR_RE = re.compile(r"(?i)\b(error|exception|traceback|failed|failure|warning|denied|not found|cannot|timeout|invalid|undefined|null)\b")
URL_RE = re.compile(r"https?://[^\s\"'<>]+")
PACKAGE_RE = re.compile(r"(?i)\b(?:install|add|require|import|from|package|module|version|v?\d+\.\d+(?:\.\d+)?)\b[^\n\r]{0,160}")


@dataclass
class FrameNote:
    path: str
    timestamp: float
    source: str
    ocr: str = ""


def warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[info] {message}", file=sys.stderr)


def have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def find_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found

    home = Path.home()
    candidates = {
        "ffmpeg": [
            home / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe",
            Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        ],
        "ffprobe": [
            home / "AppData/Local/Microsoft/WinGet/Links/ffprobe.exe",
            Path("C:/Program Files/ffmpeg/bin/ffprobe.exe"),
        ],
        "tesseract": [
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        ],
    }

    for candidate in candidates.get(name, []):
        if candidate.exists():
            return str(candidate)
    return None


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def timestamp_label(seconds: float) -> str:
    seconds = max(0, float(seconds))
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}-{minutes:02d}-{sec:02d}"
    return f"{minutes:02d}-{sec:02d}"


def timestamp_text(seconds: float) -> str:
    return timestamp_label(seconds).replace("-", ":")


def duration_text(seconds: float) -> str:
    return f"{max(0.0, float(seconds)):.3f}s"


def make_chunks(duration: float | None, chunk_minutes: float, overlap_seconds: float) -> list[tuple[float, float]]:
    if not duration or duration <= 0 or chunk_minutes <= 0:
        return []

    chunk_seconds = max(30.0, chunk_minutes * 60.0)
    overlap_seconds = max(0.0, min(overlap_seconds, chunk_seconds / 2))
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        chunks.append((start, end))
        if end >= duration:
            break
        start = max(0.0, end - overlap_seconds)
    return chunks


def in_range(seconds: float, start: float | None, end: float | None) -> bool:
    if start is not None and seconds < start:
        return False
    if end is not None and seconds > end:
        return False
    return True


def get_video_duration(video: Path) -> float | None:
    ffprobe = find_executable("ffprobe")
    if ffprobe:
        result = run([
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ])
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass

    if have_module("cv2"):
        import cv2

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frames > 0:
            return float(frames / fps)

    return None


def save_frame_cv2(video: Path, out_file: Path, seconds: float) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_file), frame))


def save_frame_ffmpeg(video: Path, out_file: Path, seconds: float) -> bool:
    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        return False
    out_file.parent.mkdir(parents=True, exist_ok=True)
    result = run([
        ffmpeg,
        "-y",
        "-ss",
        str(max(0, seconds)),
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_file),
    ])
    return result.returncode == 0 and out_file.exists()


def save_frame(video: Path, out_file: Path, seconds: float) -> bool:
    if have_module("cv2"):
        try:
            if save_frame_cv2(video, out_file, seconds):
                return True
        except Exception as exc:
            warn(f"OpenCV frame extraction failed at {seconds:.2f}s: {exc}")
    try:
        return save_frame_ffmpeg(video, out_file, seconds)
    except Exception as exc:
        warn(f"FFmpeg frame extraction failed at {seconds:.2f}s: {exc}")
        return False


def scene_midpoints(video: Path, max_frames: int) -> list[float]:
    if not have_module("scenedetect"):
        return []

    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        video_stream = open_video(str(video))
        scene_manager.detect_scenes(video_stream)
        scenes = scene_manager.get_scene_list()
        points: list[float] = []
        for start, end in scenes[:max_frames]:
            points.append((start.get_seconds() + end.get_seconds()) / 2)
        return points
    except Exception as exc:
        warn(f"PySceneDetect failed; falling back to interval sampling: {exc}")
        return []


def interval_midpoints(duration: float | None, interval: float, max_frames: int) -> list[float]:
    if duration is None or duration <= 0:
        return [0.0]
    points: list[float] = []
    current = 0.0
    while current <= duration and len(points) < max_frames:
        points.append(current)
        current += interval
    if duration > 0 and duration not in points and len(points) < max_frames:
        points.append(max(0.0, duration - 1))
    return points


def select_evenly[T](items: list[T], limit: int) -> list[T]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indexes = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indexes]


def sampled_points_for_range(start: float, end: float, interval: float, limit: int) -> list[float]:
    if limit <= 0:
        return []
    start = max(0.0, start)
    end = max(start, end)
    interval = max(0.1, interval)

    points: list[float] = []
    current = start
    while current <= end:
        points.append(current)
        current += interval

    if not points:
        points = [(start + end) / 2]
    return select_evenly(points, limit)


def chunk_interval_midpoints(
    chunk_ranges: list[tuple[float, float]],
    interval: float,
    max_frames: int,
) -> list[float]:
    if not chunk_ranges or max_frames <= 0:
        return []

    base = max_frames // len(chunk_ranges)
    remainder = max_frames % len(chunk_ranges)
    points: list[float] = []
    seen: set[float] = set()
    for index, chunk in enumerate(chunk_ranges):
        limit = base + (1 if index < remainder else 0)
        if base == 0:
            limit = 1 if index < max_frames else 0
        for point in sampled_points_for_range(chunk[0], chunk[1], interval, limit):
            key = round(point, 3)
            if key not in seen:
                seen.add(key)
                points.append(point)
    return sorted(points)


def extract_frames(
    video: Path,
    out_dir: Path,
    interval: float,
    max_frames: int,
    chunk_ranges: list[tuple[float, float]] | None = None,
    duration: float | None = None,
) -> list[FrameNote]:
    if not have_module("cv2") and not find_executable("ffmpeg"):
        warn("No OpenCV module or ffmpeg binary found; skipping keyframe extraction.")
        return []

    duration = duration if duration is not None else get_video_duration(video)
    if chunk_ranges:
        points = chunk_interval_midpoints(chunk_ranges, interval, max_frames)
        source = "chunk-interval"
    else:
        points = scene_midpoints(video, max_frames)
        source = "scene"
    if not points:
        points = interval_midpoints(duration, interval, max_frames)
        source = "interval"

    frames: list[FrameNote] = []
    for index, seconds in enumerate(points[:max_frames], start=1):
        label = timestamp_label(seconds)
        out_file = out_dir / "frames" / f"{index:03d}_{label}.jpg"
        if save_frame(video, out_file, seconds):
            frames.append(FrameNote(path=str(out_file), timestamp=seconds, source=source))
    return frames


def add_ocr(frames: list[FrameNote]) -> None:
    if not frames:
        return
    tesseract = find_executable("tesseract")
    if not tesseract or not have_module("pytesseract"):
        warn("Tesseract binary or pytesseract module missing; skipping local OCR.")
        return

    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = tesseract

    for frame in frames:
        try:
            text = pytesseract.image_to_string(frame.path).strip()
            frame.ocr = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        except Exception as exc:
            warn(f"OCR failed for {frame.path}: {exc}")


def run_whisper(video: Path, out_dir: Path, model_name: str) -> dict | None:
    if not have_module("whisper"):
        warn("openai-whisper module missing; skipping Whisper transcript.")
        return None

    try:
        ffmpeg = find_executable("ffmpeg")
        if ffmpeg:
            ffmpeg_dir = str(Path(ffmpeg).parent)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if ffmpeg_dir not in path_parts:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

        import whisper

        info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        result = model.transcribe(str(video), fp16=False, verbose=False)
        transcript_file = out_dir / "transcript_whisper.json"
        transcript_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    except Exception as exc:
        warn(f"Whisper transcription failed: {exc}")
        return None


def load_transcript(path: Path | None) -> dict | str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def transcript_excerpt(
    transcript: dict | str | None,
    max_chars: int,
    start: float | None = None,
    end: float | None = None,
) -> str:
    if transcript is None:
        return ""
    if isinstance(transcript, str):
        return transcript[:max_chars]
    segments = transcript.get("segments") or []
    if segments:
        lines = []
        for segment in segments:
            segment_start = float(segment.get("start", 0) or 0)
            segment_end = float(segment.get("end", 0) or 0)
            if end is not None and segment_start > end:
                continue
            if start is not None and segment_end < start:
                continue
            text = str(segment.get("text", "")).strip()
            if text:
                lines.append(f"{segment_start:0.1f}-{segment_end:0.1f}s: {text}")
        return "\n".join(lines)[:max_chars]
    return json.dumps(transcript, ensure_ascii=False)[:max_chars]


def frame_context(
    frames: list[FrameNote],
    max_chars: int,
    start: float | None = None,
    end: float | None = None,
) -> str:
    chunks = []
    for frame in frames:
        if not in_range(frame.timestamp, start, end):
            continue
        label = timestamp_text(frame.timestamp)
        ocr = frame.ocr.strip() if frame.ocr else "(no OCR text)"
        chunks.append(f"{label} [{frame.source}] {Path(frame.path).name}\nOCR:\n{ocr}")
    return "\n\n".join(chunks)[:max_chars]


def clean_fact_text(text: str) -> str:
    return " ".join(str(text).replace("\x00", " ").split())[:300]


def add_fact(
    facts: dict[str, list[dict[str, object]]],
    kind: str,
    timestamp: float | None,
    source: str,
    text: str,
) -> None:
    cleaned = clean_fact_text(text)
    if not cleaned:
        return
    item = {
        "timestamp": timestamp,
        "time": timestamp_text(timestamp) if timestamp is not None else None,
        "source": source,
        "text": cleaned,
    }
    if item not in facts[kind]:
        facts[kind].append(item)


def scan_fact_line(
    facts: dict[str, list[dict[str, object]]],
    timestamp: float | None,
    source: str,
    line: str,
) -> None:
    line = clean_fact_text(line)
    if not line:
        return

    if ERROR_RE.search(line):
        add_fact(facts, "errors", timestamp, source, line)
    for match in COMMAND_RE.finditer(line):
        add_fact(facts, "commands", timestamp, source, match.group(0))
    for match in PATH_RE.finditer(line):
        add_fact(facts, "paths", timestamp, source, match.group(0))
    for match in URL_RE.finditer(line):
        add_fact(facts, "urls", timestamp, source, match.group(0))
    if PACKAGE_RE.search(line):
        add_fact(facts, "packages_versions", timestamp, source, line)


def extract_coding_evidence(frames: list[FrameNote], transcript: dict | str | None) -> dict[str, object]:
    facts: dict[str, list[dict[str, object]]] = {
        "commands": [],
        "paths": [],
        "errors": [],
        "urls": [],
        "packages_versions": [],
    }

    for frame in frames:
        if not frame.ocr:
            continue
        for line in frame.ocr.splitlines():
            scan_fact_line(facts, frame.timestamp, "ocr", line)

    if isinstance(transcript, dict):
        for segment in transcript.get("segments") or []:
            timestamp = segment.get("start")
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                timestamp = None
            text = str(segment.get("text", ""))
            for line in text.splitlines() or [text]:
                scan_fact_line(facts, timestamp, "transcript", line)
    elif isinstance(transcript, str):
        for line in transcript.splitlines():
            scan_fact_line(facts, None, "transcript", line)

    return {
        "counts": {key: len(value) for key, value in facts.items()},
        "facts": facts,
    }


def evidence_context(evidence: dict[str, object] | None, max_chars: int) -> str:
    if not evidence:
        return ""
    facts = evidence.get("facts", {})
    lines = []
    for kind in ["commands", "paths", "errors", "urls", "packages_versions"]:
        values = facts.get(kind, []) if isinstance(facts, dict) else []
        if not values:
            continue
        lines.append(f"{kind}:")
        for item in values[:80]:
            time_value = item.get("time") or "unknown"
            source = item.get("source") or "unknown"
            text = item.get("text") or ""
            lines.append(f"- {time_value} [{source}] {text}")
    return "\n".join(lines)[:max_chars]


def build_prompt(
    args: argparse.Namespace,
    frames: list[FrameNote],
    transcript: dict | str | None,
    evidence: dict[str, object] | None,
    chunk: tuple[float, float] | None = None,
) -> str:
    start = chunk[0] if chunk else None
    end = chunk[1] if chunk else None
    transcript_text = transcript_excerpt(transcript, args.max_transcript_chars, start, end)
    frames_text = frame_context(frames, args.max_ocr_chars, start, end)
    evidence_text = evidence_context(evidence, args.max_evidence_chars)
    task = args.prompt or (CODING_TASK if args.coding_mode else (
        "Create a precise timestamped understanding of this video. "
        "Use visual evidence, speech evidence, and OCR/keyframe context. "
        "Do not guess when evidence is unclear."
    ))

    chunk_text = ""
    if chunk:
        chunk_text = f"\nAnalyze only this time range: {timestamp_text(chunk[0])} to {timestamp_text(chunk[1])}.\n"

    output_instruction = (
        "Return valid JSON matching the requested schema. Keep strings concise, but preserve exact commands, paths, errors, and code identifiers."
        if args.structured_json
        else "Return the requested sections in clear Markdown."
    )

    return f"""You are analyzing a video with extra preprocessing context.

Task:
{task}
{chunk_text}
Output format:
{output_instruction}

Return:
1. A concise executive summary.
2. A timestamped timeline of meaningful events.
3. Important on-screen text or UI details.
4. Any uncertainty or missing evidence.
5. A JSON block with events: timestamp, visual_evidence, speech_evidence, ocr_evidence, inference, confidence.

Transcript context:
{transcript_text if transcript_text else "(no external transcript supplied)"}

Extracted coding evidence:
{evidence_text if evidence_text else "(no extracted coding evidence supplied)"}

Keyframe/OCR context:
{frames_text if frames_text else "(no external keyframe/OCR context supplied)"}
"""


def build_merge_prompt(args: argparse.Namespace, chunk_results: list[dict[str, str]]) -> str:
    output_instruction = (
        "Return valid JSON matching the requested schema."
        if args.structured_json
        else "Return a concise Markdown report."
    )
    return f"""Merge these per-chunk coding video analyses into one final answer.

Rules:
- Preserve exact commands, filenames, errors, versions, and config keys.
- De-duplicate repeated steps from overlapping chunks.
- Prefer timestamped evidence over inference.
- Mark anything uncertain as unclear.
- {output_instruction}

Chunk analyses:
{json.dumps(chunk_results, indent=2, ensure_ascii=False)}
"""


def build_verification_prompt(
    args: argparse.Namespace,
    analysis: str,
    frames: list[FrameNote],
    transcript: dict | str | None,
    evidence: dict[str, object] | None,
) -> str:
    output_instruction = (
        "Return corrected valid JSON matching the requested schema."
        if args.structured_json
        else "Return a corrected concise Markdown report."
    )
    transcript_text = transcript_excerpt(transcript, args.max_transcript_chars)
    frames_text = frame_context(frames, args.max_ocr_chars)
    evidence_text = evidence_context(evidence, args.max_evidence_chars)
    return f"""Verify the previous analysis against the video and supplied OCR/transcript evidence.

Check for:
- missing commands, filenames, errors, stack traces, or package names
- hallucinated steps not supported by video, OCR, or transcript
- contradictory timestamps
- unclear screen text that should be marked uncertain

{output_instruction}

Transcript context:
{transcript_text if transcript_text else "(no external transcript supplied)"}

Extracted coding evidence:
{evidence_text if evidence_text else "(no extracted coding evidence supplied)"}

Keyframe/OCR context:
{frames_text if frames_text else "(no external keyframe/OCR context supplied)"}

Previous analysis:
{analysis}
"""


def media_resolution_level(args: argparse.Namespace, types: object) -> object | None:
    level = args.media_resolution
    if args.high_res:
        level = "high"
    if level == "auto":
        return None
    enum_name = {
        "low": "MEDIA_RESOLUTION_LOW",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "high": "MEDIA_RESOLUTION_HIGH",
        "ultra": "MEDIA_RESOLUTION_ULTRA_HIGH",
    }[level]
    return getattr(types.PartMediaResolutionLevel, enum_name)


def model_media_resolution(args: argparse.Namespace, types: object) -> object | None:
    level = args.media_resolution
    if args.high_res:
        level = "high"
    if level == "auto":
        return None
    enum_name = {
        "low": "MEDIA_RESOLUTION_LOW",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "high": "MEDIA_RESOLUTION_HIGH",
        "ultra": "MEDIA_RESOLUTION_HIGH",
    }[level]
    return getattr(types.MediaResolution, enum_name)


def make_generate_config(args: argparse.Namespace, types: object) -> object:
    config: dict[str, object] = {"temperature": args.temperature}
    resolution = model_media_resolution(args, types)
    if resolution is not None:
        config["media_resolution"] = resolution
    if args.structured_json:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = CODING_JSON_SCHEMA if args.coding_mode else GENERIC_JSON_SCHEMA
    if args.code_execution:
        config["tools"] = [types.Tool(code_execution=types.ToolCodeExecution())]
    return types.GenerateContentConfig(**config)


def make_video_part(
    uploaded_video: object,
    args: argparse.Namespace,
    types: object,
    chunk: tuple[float, float] | None = None,
) -> object:
    media_resolution = media_resolution_level(args, types)
    part_resolution = None
    if media_resolution is not None:
        part_resolution = types.PartMediaResolution(level=media_resolution)

    uri = getattr(uploaded_video, "uri", None)
    mime_type = getattr(uploaded_video, "mime_type", None) or getattr(uploaded_video, "mimeType", None)
    video_part = types.Part.from_uri(file_uri=uri, mime_type=mime_type, media_resolution=part_resolution)
    if args.gemini_fps or chunk:
        metadata: dict[str, object] = {}
        if args.gemini_fps:
            metadata["fps"] = args.gemini_fps
        if chunk:
            metadata["start_offset"] = duration_text(chunk[0])
            metadata["end_offset"] = duration_text(chunk[1])
        video_part.video_metadata = types.VideoMetadata(**metadata)
    return video_part


def frames_for_upload(
    frames: list[FrameNote],
    args: argparse.Namespace,
    chunk_ranges: list[tuple[float, float]] | None = None,
) -> list[FrameNote]:
    if not args.upload_keyframes:
        return []
    if not chunk_ranges:
        return select_evenly(frames, args.max_uploaded_frames)

    selected: list[FrameNote] = []
    seen: set[str] = set()
    for chunk in chunk_ranges:
        matching = [frame for frame in frames if in_range(frame.timestamp, chunk[0], chunk[1])]
        for frame in select_evenly(matching, args.max_uploaded_frames):
            if frame.path not in seen:
                seen.add(frame.path)
                selected.append(frame)
    return selected


def upload_keyframes(
    client: object,
    args: argparse.Namespace,
    frames: list[FrameNote],
    chunk_ranges: list[tuple[float, float]] | None = None,
) -> list[tuple[FrameNote, object]]:
    uploaded_frames: list[tuple[FrameNote, object]] = []

    for frame in frames_for_upload(frames, args, chunk_ranges):
        try:
            uploaded_frame = client.files.upload(file=frame.path)
            uploaded_frames.append((frame, wait_for_file(client, uploaded_frame)))
        except Exception as exc:
            warn(f"Could not upload keyframe {frame.path}: {exc}")
    return uploaded_frames


def selected_keyframes(
    uploaded_frames: list[tuple[FrameNote, object]],
    args: argparse.Namespace,
    chunk: tuple[float, float] | None = None,
) -> list[object]:
    start = chunk[0] if chunk else None
    end = chunk[1] if chunk else None
    selected = [item for item in uploaded_frames if in_range(item[0].timestamp, start, end)]
    selected = select_evenly(selected, args.max_uploaded_frames)
    parts: list[object] = []
    for frame, uploaded in selected:
        parts.append(f"Keyframe timestamp: {timestamp_text(frame.timestamp)}")
        parts.append(uploaded)
    return parts


def generate_once(
    client: object,
    args: argparse.Namespace,
    types: object,
    uploaded_video: object,
    uploaded_frames: list[tuple[FrameNote, object]],
    prompt: str,
    chunk: tuple[float, float] | None = None,
) -> str:
    contents = [make_video_part(uploaded_video, args, types, chunk)]
    contents.extend(selected_keyframes(uploaded_frames, args, chunk))
    contents.append(prompt)
    response = client.models.generate_content(
        model=args.model,
        contents=contents,
        config=make_generate_config(args, types),
    )
    return getattr(response, "text", str(response))


def upload_and_generate(
    args: argparse.Namespace,
    video: Path,
    frames: list[FrameNote],
    transcript: dict | str | None,
    evidence: dict[str, object] | None,
    out_dir: Path,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise SystemExit("Missing Gemini SDK. Install it with: python -m pip install google-genai") from exc

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini analysis.")

    client = genai.Client()
    info(f"Uploading video to Gemini: {video}")
    uploaded_video = client.files.upload(file=str(video))
    uploaded_video = wait_for_file(client, uploaded_video)

    duration = get_video_duration(video)
    chunks = make_chunks(duration, args.chunk_minutes, args.chunk_overlap_seconds)
    uploaded_frames = upload_keyframes(client, args, frames, chunks)
    if chunks:
        chunk_results: list[dict[str, str]] = []
        for index, chunk in enumerate(chunks, start=1):
            info(f"Calling Gemini for chunk {index}/{len(chunks)}: {timestamp_text(chunk[0])}-{timestamp_text(chunk[1])}")
            chunk_prompt = build_prompt(args, frames, transcript, evidence, chunk)
            result = generate_once(client, args, types, uploaded_video, uploaded_frames, chunk_prompt, chunk)
            chunk_results.append({
                "range": f"{timestamp_text(chunk[0])}-{timestamp_text(chunk[1])}",
                "analysis": result,
            })
        (out_dir / "chunk_analyses.json").write_text(json.dumps(chunk_results, indent=2, ensure_ascii=False), encoding="utf-8")

        info(f"Merging {len(chunks)} chunk analyses with Gemini model: {args.model}")
        merge_response = client.models.generate_content(
            model=args.model,
            contents=build_merge_prompt(args, chunk_results),
            config=make_generate_config(args, types),
        )
        final_result = getattr(merge_response, "text", str(merge_response))
    else:
        info(f"Calling Gemini model: {args.model}")
        final_prompt = build_prompt(args, frames, transcript, evidence)
        final_result = generate_once(client, args, types, uploaded_video, uploaded_frames, final_prompt)

    if args.verify_pass:
        info("Running verification pass")
        verification_prompt = build_verification_prompt(args, final_result, frames, transcript, evidence)
        final_result = generate_once(client, args, types, uploaded_video, uploaded_frames, verification_prompt)

    return final_result


def file_state_name(file_obj: object) -> str:
    state = getattr(file_obj, "state", None)
    name = getattr(state, "name", None)
    if name:
        return str(name)
    if state:
        return str(state)
    return ""


def wait_for_file(client: object, file_obj: object) -> object:
    name = getattr(file_obj, "name", None)
    for _ in range(90):
        state_name = file_state_name(file_obj).upper()
        if "PROCESSING" not in state_name:
            if "FAILED" in state_name:
                raise RuntimeError(f"Gemini file processing failed for {name}")
            return file_obj
        time.sleep(2)
        if name:
            file_obj = client.files.get(name=name)
    raise TimeoutError(f"Timed out waiting for Gemini file processing: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build transcript/OCR/keyframe context and send a video to Gemini.")
    parser.add_argument("video", type=Path, help="Path to a local video file.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to ./video_gemini_out/<video-name>.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--prompt", default="", help="Custom analysis prompt.")
    parser.add_argument("--deep-crunch", action="store_true", help="Opinionated high-detail coding-video mode: OCR/transcript/facts/chunks/high-res/verification.")
    parser.add_argument("--coding-mode", action="store_true", help="Use a coding-video prompt focused on commands, files, errors, and fixes.")
    parser.add_argument("--frame-interval", type=float, default=10.0, help="Seconds between fallback sampled frames.")
    parser.add_argument("--max-frames", type=int, default=24, help="Maximum extracted frames.")
    parser.add_argument("--upload-keyframes", action="store_true", help="Upload selected keyframes to Gemini in addition to the video.")
    parser.add_argument("--max-uploaded-frames", type=int, default=8, help="Maximum keyframes uploaded when --upload-keyframes is set.")
    parser.add_argument("--gemini-fps", type=float, default=None, help="Override Gemini video sampling FPS, e.g. 3 or 5 for fast coding videos.")
    parser.add_argument("--high-res", action="store_true", help="Use high media resolution for small code/terminal text.")
    parser.add_argument("--media-resolution", choices=["auto", "low", "medium", "high", "ultra"], default="auto", help="Explicit media resolution. --high-res is shorthand for high.")
    parser.add_argument("--structured-json", action="store_true", help="Ask Gemini for schema-constrained JSON output.")
    parser.add_argument("--verify-pass", action="store_true", help="Run a second Gemini pass to catch missed commands/errors and contradictions.")
    parser.add_argument("--code-execution", action="store_true", help="Enable Gemini code execution for code-heavy analysis.")
    parser.add_argument("--chunk-minutes", type=float, default=0.0, help="Analyze long videos in N-minute chunks, then merge.")
    parser.add_argument("--chunk-overlap-seconds", type=float, default=10.0, help="Overlap between Gemini chunks.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Gemini temperature.")
    parser.add_argument("--transcript", type=Path, default=None, help="Existing transcript file, JSON or text.")
    parser.add_argument("--run-whisper", action="store_true", help="Run local Whisper if installed.")
    parser.add_argument("--whisper-model", default="base", help="Whisper model for --run-whisper.")
    parser.add_argument("--skip-gemini", action="store_true", help="Only build local context files.")
    parser.add_argument("--max-transcript-chars", type=int, default=60000)
    parser.add_argument("--max-ocr-chars", type=int, default=30000)
    parser.add_argument("--max-evidence-chars", type=int, default=30000)
    return parser.parse_args()


def apply_deep_crunch_defaults(args: argparse.Namespace) -> None:
    if not args.deep_crunch:
        return

    args.coding_mode = True
    args.upload_keyframes = True
    args.run_whisper = True
    args.high_res = True
    args.structured_json = True
    args.verify_pass = True
    if args.gemini_fps is None:
        args.gemini_fps = 3.0
    if args.frame_interval == 10.0:
        args.frame_interval = 1.0
    if args.max_frames == 24:
        args.max_frames = 300
    if args.max_uploaded_frames == 8:
        args.max_uploaded_frames = 12
    if args.chunk_minutes == 0.0:
        args.chunk_minutes = 3.0
    if args.max_transcript_chars == 60000:
        args.max_transcript_chars = 100000
    if args.max_ocr_chars == 30000:
        args.max_ocr_chars = 80000
    if args.max_evidence_chars == 30000:
        args.max_evidence_chars = 80000


def main() -> int:
    args = parse_args()
    apply_deep_crunch_defaults(args)
    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    out_dir = args.out or (Path.cwd() / "video_gemini_out" / video.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video)
    chunks = make_chunks(duration, args.chunk_minutes, args.chunk_overlap_seconds)
    frames = extract_frames(video, out_dir, args.frame_interval, args.max_frames, chunks, duration)
    add_ocr(frames)

    transcript = load_transcript(args.transcript)
    if args.run_whisper:
        transcript = run_whisper(video, out_dir, args.whisper_model) or transcript

    evidence = extract_coding_evidence(frames, transcript)
    (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "evidence.txt").write_text(evidence_context(evidence, args.max_evidence_chars), encoding="utf-8")

    context = {
        "video": str(video),
        "model": args.model,
        "duration_seconds": duration,
        "chunks": [{"start": start, "end": end, "range": f"{timestamp_text(start)}-{timestamp_text(end)}"} for start, end in chunks],
        "settings": {
            "coding_mode": args.coding_mode,
            "deep_crunch": args.deep_crunch,
            "gemini_fps": args.gemini_fps,
            "high_res": args.high_res,
            "media_resolution": args.media_resolution,
            "structured_json": args.structured_json,
            "verify_pass": args.verify_pass,
            "code_execution": args.code_execution,
            "chunk_minutes": args.chunk_minutes,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
            "frame_interval": args.frame_interval,
            "max_frames": args.max_frames,
            "upload_keyframes": args.upload_keyframes,
            "max_uploaded_frames": args.max_uploaded_frames,
            "max_evidence_chars": args.max_evidence_chars,
        },
        "frames": [asdict(frame) for frame in frames],
        "evidence_counts": evidence.get("counts", {}),
        "transcript_source": str(args.transcript) if args.transcript else ("whisper" if args.run_whisper else None),
        "missing_tools": {
            "ffmpeg": not bool(find_executable("ffmpeg")),
            "ffprobe": not bool(find_executable("ffprobe")),
            "tesseract": not bool(find_executable("tesseract")),
            "opencv_python": not have_module("cv2"),
            "pyscenedetect": not have_module("scenedetect"),
            "pytesseract": not have_module("pytesseract"),
            "openai_whisper": not have_module("whisper"),
        },
    }
    (out_dir / "context.json").write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")

    prompt = build_prompt(args, frames, transcript, evidence)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    if args.skip_gemini:
        info(f"Wrote local context to {out_dir}")
        return 0

    result = upload_and_generate(args, video, frames, transcript, evidence, out_dir)
    (out_dir / "gemini_analysis.txt").write_text(result, encoding="utf-8")
    print(result)
    info(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
