import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from dotenv import load_dotenv
from qwen_asr import Qwen3ASRModel

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")


@dataclass
class Subtitle:
    """Represents a single subtitle entry."""
    index: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str


@dataclass
class Config:
    """Configuration loaded from .env file."""
    # Model settings
    model_path: str = os.getenv("MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
    aligner_path: str = os.getenv("ALIGNER_PATH", "Qwen/Qwen3-ForcedAligner-0.6B")
    device: str = os.getenv("DEVICE", "cuda:0")

    # Inference settings
    max_inference_batch_size: int = int(os.getenv("MAX_INFERENCE_BATCH_SIZE", "64"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "1024"))
    use_flash_attention: bool = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"

    # Punctuation settings
    primary_endings: set = None
    sentence_endings: set = None

    # Language settings
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "Chinese")
    pause_threshold: float = float(os.getenv("PAUSE_THRESHOLD", "0.3"))
    min_chars_before_break: int = int(os.getenv("MIN_CHARS_BEFORE_BREAK", "8"))

    # Max chars per language
    max_chars_chinese: int = int(os.getenv("MAX_CHARS_CHINESE", "25"))
    max_chars_korean: int = int(os.getenv("MAX_CHARS_KOREAN", "40"))
    max_chars_english: int = int(os.getenv("MAX_CHARS_ENGLISH", "80"))
    max_chars_default: int = int(os.getenv("MAX_CHARS_DEFAULT", "50"))

    # Duration and gap settings
    max_subtitle_duration: float = float(os.getenv("MAX_SUBTITLE_DURATION", "8.0"))  # Max seconds per subtitle
    max_pause_gap: float = float(os.getenv("MAX_PAUSE_GAP", "1.5"))  # Force split if gap > this

    def __post_init__(self):
        # Load punctuation from env
        self.primary_endings = set(os.getenv("PRIMARY_ENDINGS", "。？！.?!¿¡"))
        self.sentence_endings = set(os.getenv("SENTENCE_ENDINGS", "。？！.?!;；…，,、¿¡"))

    def get_max_chars(self, language: str) -> int:
        """Get max characters per subtitle based on language."""
        lang_lower = language.lower() if language else ""
        if "chinese" in lang_lower or "zh" in lang_lower:
            return self.max_chars_chinese
        elif "korean" in lang_lower or "ko" in lang_lower:
            return self.max_chars_korean
        elif "english" in lang_lower or "en" in lang_lower:
            return self.max_chars_english
        return self.max_chars_default


# Global config instance
CONFIG = Config()

# For backward compatibility
SENTENCE_ENDINGS = CONFIG.sentence_endings
PRIMARY_ENDINGS = CONFIG.primary_endings


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timecode format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class SentenceSegmenter(ABC):
    """Abstract base class for sentence segmentation strategies."""

    @abstractmethod
    def segment(self, timestamps, text: str, language: str) -> List[Subtitle]:
        """Segment into subtitles."""
        pass


class PauseBasedSegmenter(SentenceSegmenter):
    """
    Segmenter based on pause detection in timestamps.
    Uses pause duration and character limits to determine breaks.
    """

    def __init__(
        self,
        pause_threshold: float = 0.3,
        max_chars: int = 25,
        min_chars: int = 8
    ):
        self.pause_threshold = pause_threshold
        self.max_chars = max_chars
        self.min_chars = min_chars

    def segment(self, timestamps, text: str, language: str) -> List[Subtitle]:
        if timestamps is None or len(timestamps) == 0:
            return []

        subtitles = []
        current_text = ""
        sentence_start_time = None
        sentence_end_time = None
        subtitle_index = 1
        prev_end_time = None

        for item in timestamps:
            item_text = item.text
            start_time = item.start_time
            end_time = item.end_time

            should_break = False

            if current_text and prev_end_time is not None:
                pause_duration = start_time - prev_end_time

                if pause_duration >= self.pause_threshold and len(current_text) >= self.min_chars:
                    should_break = True

                if len(current_text) >= self.max_chars:
                    should_break = True

            if should_break and current_text.strip():
                subtitles.append(Subtitle(
                    index=subtitle_index,
                    start_time=sentence_start_time,
                    end_time=sentence_end_time,
                    text=current_text.strip()
                ))
                subtitle_index += 1
                current_text = ""
                sentence_start_time = None

            if sentence_start_time is None:
                sentence_start_time = start_time

            current_text += item_text
            sentence_end_time = end_time
            prev_end_time = end_time

        if current_text.strip():
            subtitles.append(Subtitle(
                index=subtitle_index,
                start_time=sentence_start_time,
                end_time=sentence_end_time,
                text=current_text.strip()
            ))

        return subtitles


class PunctuationBasedSegmenter(SentenceSegmenter):
    """
    Segmenter based on punctuation from transcribed text.
    Maps punctuation from full text back to timestamps for accurate timing.
    Supports: Chinese, Korean, Japanese, English, Vietnamese, Spanish, French, German.
    """

    def __init__(self, max_chars: int = None, language: str = None):
        self.language = language
        # Use language-specific max_chars from config
        if max_chars is not None:
            self.max_chars = max_chars
        elif language:
            self.max_chars = CONFIG.get_max_chars(language)
        else:
            self.max_chars = CONFIG.max_chars_default

    def segment(self, timestamps, text: str, language: str) -> List[Subtitle]:
        if timestamps is None or len(timestamps) == 0:
            return []

        # Update max_chars based on detected language
        if language and not self.language:
            self.max_chars = CONFIG.get_max_chars(language)

        # Extract sentences with punctuation from the full text
        sentences = self._split_by_punctuation(text)

        if not sentences:
            return []

        # Map sentences to timestamps
        return self._map_sentences_to_timestamps(sentences, timestamps)

    def _split_by_punctuation(self, text: str) -> List[str]:
        """Split text into sentences based on punctuation."""
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in CONFIG.primary_endings:
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            elif char in CONFIG.sentence_endings and len(current) >= self.max_chars:
                # Break at secondary punctuation if sentence is too long
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        # Handle remaining text
        if current.strip():
            sentences.append(current.strip())

        return sentences

    def _map_sentences_to_timestamps(
        self, sentences: List[str], timestamps
    ) -> List[Subtitle]:
        """
        Map sentences to timestamps by matching characters.
        Also checks for large gaps and long durations to split subtitles.

        Duration split logic:
        - If duration > max_subtitle_duration AND has punctuation: split at last punctuation
        - If duration > max_subtitle_duration AND no punctuation: keep as is
        """
        subtitles = []
        ts_index = 0
        subtitle_index = 1

        for sentence in sentences:
            if ts_index >= len(timestamps):
                break

            # Remove punctuation for matching (timestamps don't have punctuation)
            sentence_chars = [c for c in sentence if c not in CONFIG.sentence_endings and not c.isspace()]

            if not sentence_chars:
                continue

            # Collect all timestamp items for this sentence first
            sentence_ts_items = []
            temp_ts_index = ts_index
            chars_matched = 0

            while temp_ts_index < len(timestamps) and chars_matched < len(sentence_chars):
                ts_item = timestamps[temp_ts_index]
                ts_text = ts_item.text.strip()

                if ts_text and ts_text not in CONFIG.sentence_endings:
                    chars_matched += len(ts_text)

                sentence_ts_items.append(ts_item)
                temp_ts_index += 1

            # Update ts_index for next sentence
            ts_index = temp_ts_index

            if not sentence_ts_items:
                continue

            # Now process this sentence with gap and duration checks
            current_text = ""
            segment_start_time = sentence_ts_items[0].start_time
            prev_end_time = segment_start_time
            last_punct_index = -1  # Track last punctuation position in current_text
            last_punct_end_time = None

            for ts_item in sentence_ts_items:
                ts_text = ts_item.text.strip()

                # Check for large gap - force split
                gap = ts_item.start_time - prev_end_time
                if current_text and gap > CONFIG.max_pause_gap:
                    subtitles.append(Subtitle(
                        index=subtitle_index,
                        start_time=segment_start_time,
                        end_time=prev_end_time,
                        text=current_text.strip()
                    ))
                    subtitle_index += 1
                    current_text = ""
                    segment_start_time = ts_item.start_time
                    last_punct_index = -1
                    last_punct_end_time = None

                # Check for long duration
                current_duration = ts_item.end_time - segment_start_time
                if current_text and current_duration > CONFIG.max_subtitle_duration:
                    # Only split if we have punctuation in current_text
                    if last_punct_index > 0 and last_punct_end_time is not None:
                        # Split at last punctuation position
                        text_before_punct = current_text[:last_punct_index + 1].strip()
                        text_after_punct = current_text[last_punct_index + 1:].strip()

                        subtitles.append(Subtitle(
                            index=subtitle_index,
                            start_time=segment_start_time,
                            end_time=last_punct_end_time,
                            text=text_before_punct
                        ))
                        subtitle_index += 1

                        # Continue with remaining text
                        current_text = text_after_punct
                        segment_start_time = last_punct_end_time
                        last_punct_index = -1
                        last_punct_end_time = None
                    # If no punctuation, keep accumulating (don't split mid-word)

                # Add current text
                if ts_text and ts_text not in CONFIG.sentence_endings:
                    current_text += ts_text

                    # Check if this char is punctuation (for tracking split point)
                    # Look at original sentence to find punctuation after this position
                    text_len = len(current_text)
                    if text_len > 0:
                        # Find corresponding position in original sentence
                        orig_pos = 0
                        clean_count = 0
                        for c in sentence:
                            if c not in CONFIG.sentence_endings and not c.isspace():
                                clean_count += 1
                            if clean_count == text_len:
                                # Check if next char is punctuation
                                if orig_pos + 1 < len(sentence) and sentence[orig_pos + 1] in CONFIG.sentence_endings:
                                    current_text += sentence[orig_pos + 1]  # Add punctuation
                                    last_punct_index = len(current_text) - 1
                                    last_punct_end_time = ts_item.end_time
                                break
                            orig_pos += 1

                prev_end_time = ts_item.end_time

            # Add remaining text from this sentence
            if current_text.strip():
                # Find matching punctuation from original sentence
                remaining_punct = ""
                if sentence and sentence[-1] in CONFIG.sentence_endings:
                    # Only add if not already in current_text
                    if not current_text.endswith(sentence[-1]):
                        remaining_punct = sentence[-1]

                subtitles.append(Subtitle(
                    index=subtitle_index,
                    start_time=segment_start_time,
                    end_time=prev_end_time,
                    text=current_text.strip() + remaining_punct
                ))
                subtitle_index += 1

        return subtitles


def generate_srt_content(subtitles: List[Subtitle]) -> str:
    """
    Generate SRT format content from subtitles.

    Args:
        subtitles: List of Subtitle objects

    Returns:
        SRT formatted string
    """
    srt_lines = []

    for sub in subtitles:
        srt_lines.append(str(sub.index))
        srt_lines.append(f"{format_srt_time(sub.start_time)} --> {format_srt_time(sub.end_time)}")
        srt_lines.append(sub.text)
        srt_lines.append("")  # Empty line between entries

    return "\n".join(srt_lines)


def save_srt_file(srt_content: str, output_path: str) -> None:
    """
    Save SRT content to file.

    Args:
        srt_content: SRT formatted string
        output_path: Output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"SRT file saved to: {output_path}")


def transcribe_audio_to_srt(
    audio_path: str,
    model: Qwen3ASRModel,
    language: Optional[str] = "Chinese",
    output_path: Optional[str] = None,
    segmenter: Optional[SentenceSegmenter] = None,
) -> str:
    """
    Transcribe audio file and generate SRT subtitle file.

    Args:
        audio_path: Path to the audio/video file
        model: Initialized Qwen3ASRModel
        language: Language for transcription (e.g., "Chinese", "English")
        output_path: Output SRT file path (default: same as audio with .srt extension)
        segmenter: Segmentation strategy (default: PunctuationBasedSegmenter)

    Returns:
        Path to the generated SRT file
    """
    if segmenter is None:
        segmenter = PunctuationBasedSegmenter()

    # Perform ASR transcription with timestamps
    print(f"Transcribing: {audio_path}")
    results = model.transcribe(
        audio=audio_path,
        language=language,
        return_time_stamps=True,
        verbose=True
    )

    if not results:
        raise ValueError("No transcription results returned")

    result = results[0]
    detected_language = result.language or language or "Chinese"

    print(f"Detected language: {detected_language}")
    print(f"Transcribed text: {result.text[:100]}..." if len(result.text) > 100 else f"Transcribed text: {result.text}")
    """
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(result.text)
    with open("timestamps.txt", "w", encoding="utf-8") as f:
        for ts in result.time_stamps:
            f.write(f"{ts.start_time} {ts.end_time} {ts.text}\n")
    """
    # Segment into sentences
    subtitles = segmenter.segment(result.time_stamps, result.text, detected_language)
    print(f"Generated {len(subtitles)} subtitle entries")

    # Generate SRT content
    srt_content = generate_srt_content(subtitles)

    # Determine output path
    if output_path is None:
        audio_path_obj = Path(audio_path)
        output_path = str(audio_path_obj.with_suffix(".srt"))

    # Save SRT file
    save_srt_file(srt_content, output_path)

    return output_path


def create_model(
    model_path: str = None,
    aligner_path: str = None,
    device: str = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3ASRModel:
    """
    Initialize the Qwen3 ASR model with forced aligner.
    Settings loaded from .env file, can be overridden by parameters.
    Optimized for RTX 4090 (24GB VRAM).
    """
    # Use CONFIG values as defaults
    model_path = model_path or CONFIG.model_path
    aligner_path = aligner_path or CONFIG.aligner_path
    device = device or CONFIG.device

    print(f"Loading ASR model: {model_path}")
    print(f"Loading Forced Aligner: {aligner_path}")

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA is not available"
    print(f"Using device: {device}")

    # Build kwargs
    model_kwargs = dict(
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=CONFIG.max_inference_batch_size,
        max_new_tokens=CONFIG.max_new_tokens,
        forced_aligner=aligner_path,
        forced_aligner_kwargs=dict(
            dtype=dtype,
            device_map=device,
        ),
    )

    # Add flash attention if enabled
    if CONFIG.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = "flash_attention_2"

    model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)

    print("Model loaded successfully")
    print(f"Model device: {next(model.model.parameters()).device}")
    print(f"Flash Attention: {'enabled' if CONFIG.use_flash_attention else 'disabled'}")
    print(f"Max batch size: {CONFIG.max_inference_batch_size}")
    print(f"Max new tokens: {CONFIG.max_new_tokens}")
    return model


def main():
    # Initialize model (settings from .env)
    model = create_model()
    segmenter = PunctuationBasedSegmenter()

    # Read folders from .env or fallback to config.txt
    config_file = os.getenv("CONFIG_FILE", "config.txt")

    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        print("Please create a config.txt file with folder paths, one per line.")
        return

    with open(config_file, "r", encoding='utf-8') as f:
        folders = [line.strip() for line in f.readlines() if line.strip() and os.path.exists(line.strip())]

    if not folders:
        print("No valid folders found in config file.")
        return

    print(f"Found {len(folders)} folder(s) to process")

    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        for file in os.listdir(folder):
            if file.endswith(".mp4") or file.endswith(".wav") or file.endswith(".mp3"):
                srt_path = os.path.join(folder, file.replace(".mp4", ".srt")).replace(".mp3", ".srt").replace(".wav", ".srt")
                if os.path.exists(srt_path):
                    print(f"  Skipping {file} (SRT exists)")
                    continue

                output_path = transcribe_audio_to_srt(
                    audio_path=os.path.join(folder, file),
                    model=model,
                    segmenter=segmenter,
                )

                print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()