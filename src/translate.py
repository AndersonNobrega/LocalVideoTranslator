import gc
import os
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List, Tuple

import torch
import whisperx
from dotenv import load_dotenv
from pydub import AudioSegment

from model import llm_translation
from utils.constants import ROOT_PATH

load_dotenv()


def get_args() -> dict:
    """
    Parse command-line arguments for audio processing.

    Returns:
        dict: Dictionary with keys: file_path, translate, subject, save_dir.
    """
    parser = ArgumentParser(
        allow_abbrev=False,
        description="Transcribe audio using WhisperX and generate SRT subtitles.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file_path", type=str, required=True, help="Path to the audio file to process.")
    parser.add_argument("--translate", action="store_true", help="Generate translated subtitles using llm_translation.")
    parser.add_argument("--subject", action="store_true", help="Identify speakers as 'Subject 1', 'Subject 2', etc.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save generated SRT files.")
    return vars(parser.parse_args())


def seconds_to_srt_time(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time formatted as HH:MM:SS,mmm.
    """
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def transcribe_audio(model: object, audio_file: str, batch_size: int) -> Tuple[dict, object]:
    """
    Transcribe an audio file using the provided WhisperX model.

    Args:
        model (object): Loaded WhisperX model.
        audio_file (str): Path to the audio file.
        batch_size (int): Batch size for transcription.

    Returns:
        tuple: (transcription_result, loaded_audio)
    """
    audio = whisperx.load_audio(audio_file)
    return model.transcribe(audio, batch_size=batch_size), audio


def clear_cuda_memory(model: object = None) -> None:
    """
    Clear CUDA memory.

    Args:
        model (object, optional): Model to delete from memory.
    """
    if model:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def format_time(seconds: float) -> str:
    """
    Convert seconds to SRT time format (HH:MM:SS,mmm).

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time.
    """
    millis = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes = (seconds // 60) % 60
    hours = seconds // 3600
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def format_srt_content(segments: List[dict], include_speaker: bool) -> str:
    """
    Format SRT content from transcription segments.

    Args:
        segments (list): List of segments with start, end, text, and optionally speaker.
        include_speaker (bool): If True, prepend the speaker label.

    Returns:
        str: The SRT formatted content.
    """
    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        text = seg["text"]
        if include_speaker:
            speaker = seg.get("speaker", "Unknown")
            text = f"{speaker}: {text}"
        srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_lines)


def generate_srt_content(segments: List[dict], translate: bool, include_speaker: bool = False):
    """
    Generate SRT content from segments.

    Args:
        segments (list): Transcribed segments.
        translate (bool): If True, generate a translated SRT.
        include_speaker (bool): If True, include speaker labels in the SRT.

    Returns:
        tuple: (translated_srt, original_srt) if translate is True; otherwise (original_srt,)
    """
    original_srt = format_srt_content(segments, include_speaker)
    if translate:
        translated_segments = [{**seg, "text": seg.get("translated_text", seg["text"])} for seg in segments]
        translated_srt = format_srt_content(translated_segments, include_speaker)
        return translated_srt, original_srt
    return (original_srt,)


def save_srt_file(srt_content: str, file_name: str) -> None:
    """
    Save SRT content to a file.

    Args:
        srt_content (str): The SRT content.
        file_name (str): Output file name.
    """
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(srt_content)


def split_audio(file_path: str, segment_duration_ms: int = 300000) -> List[Tuple[str, float]]:
    """
    Split a large audio file into segments of a given duration.

    Args:
        file_path (str): Path to the audio file.
        segment_duration_ms (int): Duration per segment in milliseconds (default: 300,000 ms = 5 minutes).

    Returns:
        list: List of tuples (temp_file_path, start_offset_in_seconds).
    """
    audio = AudioSegment.from_file(file_path)
    segments = []
    num_segments = (len(audio) // segment_duration_ms) + (1 if len(audio) % segment_duration_ms > 0 else 0)
    for i in range(num_segments):
        start_ms = i * segment_duration_ms
        end_ms = min((i + 1) * segment_duration_ms, len(audio))
        segment_audio = audio[start_ms:end_ms]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        segment_audio.export(temp_file.name, format="wav")
        segments.append((temp_file.name, start_ms / 1000.0))
    return segments


def translate_file(
    file_path: str, save_dir: str = None, translate_flag: bool = False, subject_flag: bool = False
) -> None:
    """
    Transcribe, align, diarize, and generate SRT files from an audio file.
    For large files, the audio is split into 5‑minute segments before processing.
    Optionally, speaker labels and translated subtitles are generated.

    Args:
        file_path (str): Path to the audio file.
        save_dir (str, optional): Directory to save SRT files.
        translate_flag (bool): If True, generate translated subtitles.
        subject_flag (bool): If True, identify speakers as 'Subject 1', 'Subject 2', etc.
    """
    device = "cuda"
    batch_size = 8
    compute_type = "float16"

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    segments_files = split_audio(file_path, segment_duration_ms=300000 * 16)
    all_transcribed_segments = []

    for seg_file, offset in segments_files:
        audio_seg = whisperx.load_audio(seg_file)
        result_seg = model.transcribe(audio_seg, batch_size=batch_size)
        for seg in result_seg["segments"]:
            seg["start"] += offset
            seg["end"] += offset
        all_transcribed_segments.extend(result_seg["segments"])
        os.remove(seg_file)
    clear_cuda_memory(model)

    model_a, metadata = whisperx.load_align_model(language_code=result_seg["language"], device=device)
    full_audio = whisperx.load_audio(file_path)
    result_aligned = whisperx.align(
        all_transcribed_segments, model_a, metadata, full_audio, device, return_char_alignments=False
    )
    clear_cuda_memory(model_a)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
    diarize_segments = diarize_model(full_audio)
    clear_cuda_memory(diarize_model)

    result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned)

    if translate_flag:
        texts = [seg["text"].strip() for seg in result_final["segments"]]
        translation_result = llm_translation(texts)
        for seg, trans in zip(result_final["segments"], translation_result.get("translations", [])):
            seg["translated_text"] = trans

    srt_contents = generate_srt_content(result_final["segments"], translate_flag, include_speaker=subject_flag)

    save_original_srt_path = "original_subtitles.srt"
    save_translated_srt_path = "translated_subtitles.srt"
    if save_dir:
        save_original_srt_path = os.path.join(save_dir, save_original_srt_path)
        save_translated_srt_path = os.path.join(save_dir, save_translated_srt_path)
    else:
        save_original_srt_path = os.path.join(ROOT_PATH, save_original_srt_path)
        save_translated_srt_path = os.path.join(ROOT_PATH, save_translated_srt_path)

    if translate_flag:
        save_srt_file(srt_contents[0], save_translated_srt_path)
        save_srt_file(srt_contents[1], save_original_srt_path)
    else:
        save_srt_file(srt_contents[0], save_original_srt_path)

    print("SRT files generated successfully.")


def main() -> None:
    args = get_args()
    translate_file(
        file_path=args["file_path"],
        save_dir=args["save_dir"],
        translate_flag=args["translate"],
        subject_flag=args["subject"],
    )


if __name__ == "__main__":
    main()
