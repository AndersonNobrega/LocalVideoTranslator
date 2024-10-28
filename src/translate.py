import gc
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Tuple

import requests
import torch
import whisperx
from dotenv import load_dotenv

from .model import llm_translation

load_dotenv()


def get_args():
    """
    Parse command-line arguments for audio file processing.

    Returns:
        dict: A dictionary containing the parsed command-line arguments.
              The key will be the argument name, and the value will be
              the corresponding input provided by the user.

    Raises:
        SystemExit: This exception is raised if the required argument
                     is not provided, resulting in an error message
                     and termination of the program.
    """
    parser = ArgumentParser(allow_abbrev=False, description="", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the Audio file that will be processed.", default=None
    )

    return vars(parser.parse_args())


def seconds_to_srt_time(seconds: float) -> str:
    """
    Convert a time duration in seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The formatted time string in SRT format (HH:MM:SS,mmm).
    """
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def translate_content(content: str) -> dict:
    """
    Send a request to a local translation service to translate the given content.

    Args:
        content (str): The text content to be translated.

    Returns:
        dict: The JSON response from the translation service containing the translated text.

    Raises:
        requests.exceptions.RequestException: If the request to the translation service fails.
    """
    url = "http://localhost:5000/translate"
    payload = {"q": content, "source": "auto", "target": "pt"}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response.json()


def transcribe_audio(model: object, audio_file: str, batch_size: int) -> tuple:
    """
    Transcribe the audio file using the specified model.

    Args:
        model (object): The loaded WhisperX model.
        audio_file (str): The path to the audio file to transcribe.
        batch_size (int): The number of audio samples to process in a single batch.

    Returns:
        tuple: A tuple containing the transcription result and the loaded audio data.
    """
    audio = whisperx.load_audio(audio_file)
    return model.transcribe(audio, batch_size=batch_size), audio


def clear_cuda_memory(model: object = None) -> None:
    """
    Clear CUDA memory to free up resources.

    Args:
        model (object, optional): The model to be deleted from memory. Defaults to None.
    """
    if model:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def generate_srt_content(segments: list) -> Tuple[str]:
    """
    Generate SRT formatted content from the transcribed segments.

    Args:
        segments (list): A list of transcribed segments with start and end times.

    Returns:
        str: The generated SRT content.
    """
    translated_srt_content = ""
    original_srt_content = ""
    text_to_translate = [segment["text"].strip() for segment in segments]
    translated_text = llm_translation(text_to_translate)

    for idx, segment in enumerate(segments, start=1):
        start_time = seconds_to_srt_time(segment["start"])
        end_time = seconds_to_srt_time(segment["end"])

        translated_srt_content += f"{idx}\n{start_time} --> {end_time}\n{translated_text['translations'][idx-1]}\n\n"
        original_srt_content += f"{idx}\n{start_time} --> {end_time}\n{text_to_translate[idx-1]}\n\n"
    return translated_srt_content, original_srt_content


def save_srt_file(srt_content: str, file_name: str) -> None:
    """
    Save the generated SRT content to a file.

    Args:
        srt_content (str): The SRT content to save.
        file_name (str): The name of the file where the content will be saved.
    """
    with open(file_name, "w") as file:
        file.write(srt_content)


def translate(file_path: str, save_dir: str = None) -> None:
    """
    Execute the audio transcription, alignment, and SRT generation process for a given audio file.

    Args:
        file_path (str): The path to the audio file to be transcribed and aligned.

    Returns:
        None: This function does not return a value but generates an SRT file
        containing the subtitles based on the transcribed and aligned audio.

    Raises:
        Exception: May raise exceptions related to model loading, audio processing,
        or file operations if errors occur during execution.
    """
    device = "cuda"
    batch_size = 8
    compute_type = "float16"

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    clear_cuda_memory(model)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    clear_cuda_memory(model_a)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
    diarize_segments = diarize_model(audio)
    clear_cuda_memory(diarize_model)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    translated_srt_content, original_srt_content = generate_srt_content(result["segments"])
    save_translated_str_path = "translated_subtitles.srt"
    save_original_str_path = "original_subtitles.srt"
    if save_dir is not None:
        save_translated_str_path = save_dir + "/" + save_translated_str_path
        save_original_str_path = save_dir + "/" + save_original_str_path
    save_srt_file(translated_srt_content, save_translated_str_path)
    save_srt_file(original_srt_content, save_original_str_path)

    print("SRT files generated successfully.")


def main():
    args = get_args()
    translate(args["file_path"])


if __name__ == "__main__":
    main()
