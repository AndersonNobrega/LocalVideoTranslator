import gc
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import requests
import torch
import whisperx
from dotenv import load_dotenv

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
    url = "https://localhost:5000/translate"
    payload = {"q": content, "source": "auto", "target": "en"}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response.json()


def load_model(model_name: str, device: str, compute_type: str) -> object:
    """
    Load the WhisperX model.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to use for loading the model (e.g., 'cuda' or 'cpu').
        compute_type (str): The computation type to use (e.g., 'float16').

    Returns:
        object: The loaded WhisperX model.
    """
    return whisperx.load_model(model_name, device, compute_type=compute_type)


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


def load_alignment_model(language_code: str, device: str) -> object:
    """
    Load the alignment model for the specified language.

    Args:
        language_code (str): The language code for the model (e.g., 'en' for English).
        device (str): The device to use for loading the model.

    Returns:
        object: The loaded alignment model.
    """
    return whisperx.load_align_model(language_code=language_code, device=device)


def align_segments(model_a: object, result: dict, audio: object, device: str) -> dict:
    """
    Align the transcribed segments using the alignment model.

    Args:
        model_a (object): The loaded alignment model.
        result (dict): The transcription result containing segments.
        audio (object): The audio data for alignment.
        device (str): The device to use for processing.

    Returns:
        dict: The aligned segments.
    """
    metadata = model_a.metadata
    return whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


def generate_srt_content(segments: list) -> str:
    """
    Generate SRT formatted content from the transcribed segments.

    Args:
        segments (list): A list of transcribed segments with start and end times.

    Returns:
        str: The generated SRT content.
    """
    srt_content = ""
    for idx, segment in enumerate(segments, start=1):
        start_time = seconds_to_srt_time(segment["start"])
        end_time = seconds_to_srt_time(segment["end"])
        text = segment["text"].strip()
        translated_text = translate_content(text)

        srt_content += f"{idx}\n{start_time} --> {end_time}\n{translated_text}\n\n"
    return srt_content


def save_srt_file(srt_content: str, file_name: str) -> None:
    """
    Save the generated SRT content to a file.

    Args:
        srt_content (str): The SRT content to save.
        file_name (str): The name of the file where the content will be saved.
    """
    with open(file_name, "w") as file:
        file.write(srt_content)


def translate(file_path: str) -> None:
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

    model = load_model("large-v2", device, compute_type)
    result, audio = transcribe_audio(model, file_path, batch_size)
    clear_cuda_memory(model)

    model_a = load_alignment_model(result["language"], device)
    result = align_segments(model_a, result, audio, device)
    clear_cuda_memory(model_a)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    srt_content = generate_srt_content(result["segments"])
    save_srt_file(srt_content, "subtitles.srt")

    print("SRT file generated successfully.")


def main():
    args = get_args()
    translate(args["file_path"])


if __name__ == "__main__":
    main()
