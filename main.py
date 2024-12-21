from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from config import LANG_CODES, MODEL_PATH
from model import load_model
from audio_processing import process_audio
from translation_model import translation_model
import os
from typing import Annotated
app = FastAPI()

# Load ASR model
asr_model = load_model(MODEL_PATH)

# Directory to save PDFs
PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio_or_text(
    language: Annotated[str, Form()] = "hi",  # Move non-default argument first
    file: UploadFile = File(None),           # Default argument follows
    text: Annotated[str, Form()] = None
):
    """
    Handles audio file upload, transcribes the audio using ASR,
    translates transcription or Indic text to English.
    """
    try:
        if file:  # Handle audio file transcription
            # Save uploaded file temporarily
            file_location = f"temp/{file.filename}"
            with open(file_location, "wb") as f:
                f.write(await file.read())

            # Step 1: Transcribe the audio file
            transcription = process_audio(file_location, asr_model, language)

            # Step 2: Translate the transcription to English
            translations = translation_model.translate([transcription], language, "en")

            # Remove the temporary file
            os.remove(file_location)

            # Return both transcription and translation
            return JSONResponse(content={
                "transcription": transcription,
                "translation": translations[0]
            })

        elif text:  # Handle Indic text translation
            # Translate the text to English
            translations = translation_model.translate([text], language, "en")
            return JSONResponse(content={"translation": translations[0]})

        else:
            return JSONResponse(content={"error": "No input provided. Send audio or text."}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads and saves a PDF file for future use.
    """
    try:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(content={"error": "Only PDF files are allowed."}, status_code=400)

        file_location = os.path.join(PDF_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        return JSONResponse(content={"message": f"File {file.filename} uploaded successfully.",
                                      "path": file_location})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
