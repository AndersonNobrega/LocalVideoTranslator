import os
import socket

from flask import Flask, after_this_request, jsonify, render_template, request, send_file

from src.translate import translate
from src.utils.files import cleanup_upload_folder, create_zip_file, save_uploaded_file

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Flask to save uploaded files
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def get_ipv4_address():
    # Connect to an external address to get the local IP (does not actually send data)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Connecting to an external address (no data is sent)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception as e:
            print(f"Error: {e}")
            return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the uploads folder
    file_path = save_uploaded_file(file, app.config["UPLOAD_FOLDER"])

    # Perform translation processing
    translate(file_path, app.config["UPLOAD_FOLDER"])

    # Prepare files for download
    srt_files = ["translated_subtitles.srt", "original_subtitles.srt"]
    zip_filename = "subtitles.zip"
    zip_filepath = create_zip_file(srt_files, zip_filename, app.config["UPLOAD_FOLDER"])

    @after_this_request
    def cleanup(response):
        cleanup_upload_folder(app.config["UPLOAD_FOLDER"])
        return response

    return send_file(zip_filepath, as_attachment=True)


if __name__ == "__main__":
    ipv4_address = get_ipv4_address()
    app.run(host=ipv4_address, port=8000)
