import os
import re
import json
import uuid
import functions_framework
from flask import jsonify, Request
from google.cloud import storage
from google.auth.transport.requests import Request as AuthRequest
import google.auth
import requests

PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "edii-ocr")
LOCATION = os.environ.get("LOCATION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")

def get_access_token():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(AuthRequest())
    return credentials.token

def clean_json_text(text: str) -> str:
    return re.sub(r"```json|```", "", text).strip()

def upload_to_gcs(file_obj, destination_blob_name: str) -> str:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_file(file_obj)
    return f"gs://{BUCKET_NAME}/{destination_blob_name}"

@functions_framework.http
def pdfExtractorStream(request: Request):
    if request.method != "POST":
        return jsonify({"error": "Use POST method"}), 405

    if 'file' not in request.files:
        return jsonify({"error": "File 'file' tidak ditemukan dalam form-data"}), 400

    file = request.files['file']

    prompt = request.form.get("prompt", "Extract dokumen ini")
    system_instruction = request.form.get(
        "system_instruction",
        "Keluaran HARUS berupa 1 objek JSON valid. Gunakan nilai kosong \"\" jika data tidak ditemukan."
    )

    new_name = f"{uuid.uuid4().hex}_{file.filename}"
    try:
        file.stream.seek(0)
    except Exception:
        pass

    try:
        gcs_uri = upload_to_gcs(file.stream, new_name)
    except Exception as e:
        return jsonify({"error": "GCS upload failed", "details": str(e)}), 500

    endpoint = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}:streamGenerateContent"
    )
    payload = {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {"text": system_instruction}
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "file_data": {
                            "mimeType": file.mimetype or "application/pdf",
                            "file_uri": gcs_uri
                        }
                    }
                ]
            }
        ],

        "generationConfig": {
            "temperature": 0.0,
            "topP": 0.95,
            "maxOutputTokens": 65535,
            "responseMimeType": "application/json"
        }
    }

    try:
        access_token = get_access_token()
    except Exception as e:
        return jsonify({"error": "Failed to get access token", "details": str(e)}), 500

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, stream=True)
    except Exception as e:
        return jsonify({"error": "Request to Vertex AI failed", "details": str(e)}), 500

    if resp.status_code != 200:
        try:
            err_text = resp.text
        except Exception:
            err_text = "<no response>"
        return jsonify({
            "error": "Vertex AI error",
            "status": resp.status_code,
            "details": err_text
        }), 500

    accumulated_text = ""
    merged_parts_text = ""
    
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                obj = json.loads(line)

                cand = obj.get("candidates", [])
                if cand:
                    part = (
                        cand[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    if part:
                        accumulated_text += part
            except json.JSONDecodeError:

                accumulated_text += line
    except Exception as e:
        return jsonify({"error": "Error reading stream", "details": str(e)}), 500

    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                obj = json.loads(line)
            except:
                continue

            if "candidates" in obj:
                candidates = obj.get("candidates", [])
                if not candidates:
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        merged_parts_text += p["text"]

    except Exception as e:
        return jsonify({"error": "Error reading stream", "details": str(e)}), 500

    clean_text = clean_json_text(accumulated_text)

    try:
        json_output = json.loads(clean_text)
    except Exception as e:
        return jsonify({
            "error": "Model did not return valid JSON",
            "clean_text": clean_text,
            "raw_text": accumulated_text,
            "merged_parts_text": merged_parts_text,
            "exception": str(e)
        }), 500

    return jsonify(json_output)
