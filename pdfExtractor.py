import base64
import functions_framework
from flask import jsonify
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests
import os
import json
import re
import jwt

PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = "asia-southeast1"
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
PUBLIC_KEY_PATH = "/secrets/jwt-public.pem"
ISSUER = "external-backend"
AUDIENCE = "cloud-run"

def get_access_token():
    import google.auth
    credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token

def clean_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return text

@functions_framework.http
def pdfExtractor(request):
    prompt = request.form.get("prompt", "Extract dokumen ini")
    public_key = os.environ.get("jwt_public_key")

    auth = request.headers.get("Authorization")
    token = auth.replace("Bearer ", "")

    if not auth:
        return jsonify({"error": f"Missing Authorization header {token}" }), 401

    if request.method != "POST":
        return jsonify({"error": "Use POST method"}), 405

    if prompt == "--ping--":
        return jsonify({"message": "ping received"}), 200
        
    # Authorization
    try:
        jwt_payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
        )
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "File 'file' tidak ditemukan dalam form-data"}), 400

    file = request.files['file']

    file_bytes = file.read()
    file_b64 = base64.b64encode(file_bytes).decode("utf-8")

    mime_type = file.mimetype or "application/pdf"

    system_instruction = request.form.get("system_instruction", "Keluaran HARUS berupa **satu objek JSON valid**. Gunakan nilai kosong "" jika data tidak ditemukan.")

    endpoint = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_NAME}:generateContent"

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
                      "inline_data": {
                          "mimeType": mime_type,
                          "data": file_b64
                      }
                  }
              ]
          }
      ],
      "generationConfig": {
        "temperature": 0.0,
        "top_p": 1.0,
        "candidateCount": 1,
        "seed": 42,
        "maxOutputTokens": 20480,
        "thinking_config": {
            "thinking_budget": 0
        },
        "responseMimeType": "application/json"
      }
    }

    access_token = get_access_token()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint, json=payload, headers=headers)

    if response.status_code != 200:
        return jsonify({
            "error": "Vertex AI error",
            "status": response.status_code,
            "details": response.text
        }), 500

    data = response.json()

    try:
        text_output = data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return jsonify({"error": "Invalid response from model", "raw": data}), 500

    clean_text = clean_json(text_output)

    try:
        json_output = json.loads(clean_text)
    except Exception as e:
        return jsonify({
            "error": "Model did not return valid JSON",
            "clean_text": clean_text,
            "raw_text": text_output,
            "exception": str(e)
        }), 500

    return jsonify(json_output)
