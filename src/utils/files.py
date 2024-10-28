import os
import zipfile

from werkzeug.datastructures import FileStorage


def save_uploaded_file(file: FileStorage, folder: str) -> str:
    """Save the uploaded file to the specified uploads folder.

    Args:
        file (FileStorage): The file to be uploaded.
        folder (str): The directory where the file will be saved.

    Returns:
        str: The path of the saved file.
    """
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)
    return file_path


def create_zip_file(srt_files: list[str], zip_filename: str, folder: str) -> str:
    """Create a zip file from the specified SRT files.

    Args:
        srt_files (list[str]): A list of SRT file names to include in the zip.
        zip_filename (str): The name of the output zip file.
        folder (str): The directory containing the SRT files.

    Returns:
        str: The path of the created zip file.
    """
    zip_filepath = os.path.join(folder, zip_filename)
    with zipfile.ZipFile(zip_filepath, "w") as zipf:
        for file_name in srt_files:
            file_path = os.path.join(folder, file_name)
            zipf.write(file_path, arcname=file_name)
    return zip_filepath


def cleanup_upload_folder(folder: str) -> None:
    """Delete all files in the specified upload folder.

    Args:
        folder (str): The directory to clean up.

    Returns:
        None
    """
    for filename in os.listdir(folder):
        file_to_delete = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_to_delete):
                os.remove(file_to_delete)
        except Exception as e:
            print(f"Error deleting file {file_to_delete}: {e}")
