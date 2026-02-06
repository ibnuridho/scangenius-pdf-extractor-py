import base64
import functions_framework
from flask import jsonify
from google import genai
import os
import json
import re

MODEL_NAME = "gemini-2.5-flash-lite"

def clean_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return text

@functions_framework.http
def pdfExtractorGenAI(request):
    if request.method != "POST":
        return jsonify({"error": "Use POST method"}), 405

    if 'file' not in request.files:
        return jsonify({"error": "File 'file' tidak ditemukan dalam form-data"}), 400

    file = request.files['file']
    prompt = request.form.get("prompt", "Extract dokumen ini")

    system_instruction = request.form.get(
        "system_instruction",
        "Keluaran HARUS berupa 1 objek JSON valid. Gunakan nilai kosong '' jika data tidak ditemukan."
    )

    file_bytes = file.read()
    file_b64 = base64.b64encode(file_bytes).decode("utf-8")
    mime_type = file.mimetype or "application/pdf"

    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_CLOUD_API_KEY is not set"}), 500

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "system",
                    "parts": [{"text": system_instruction}]
                },
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": file_b64,
                            }
                        }
                    ]
                }
            ],
            config={
                "temperature": 0.1,
                "top_p": 0.95,
                "max_output_tokens": 65535,
                "response_mime_type": "application/json"
            }
        )
    except Exception as e:
        return jsonify({"error": "Google AI API Error", "details": str(e)}), 500

    try:
        text_output = response.text
    except:
        return jsonify({"error": "Invalid API response", "raw": str(response)}), 500

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
