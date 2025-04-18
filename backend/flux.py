from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gradio_client import Client
import os
import time
import zipfile
import shutil
import re

app = Flask(__name__)

# Autoriser uniquement le frontend en localhost:3000
CORS(app, resources={
    r"/llm": {
        "origins": "http://localhost:3000",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Répertoire de base pour stocker les images
BASE_DIR = r"C:\Users\omaim\Downloads\test\Flux\backend"
CLASS1_DIR = os.path.join(BASE_DIR, "Dataset")

# Fonction pour nettoyer les noms de fichiers
def clean_filename(name):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '\n', '\r', '\t']
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name

# Génération de l’image via Gradio
def generate_image_from_prompt(class_name, prompt, image_id):
    try:
        client = Client("https://be0e17d61cd7b47078.gradio.live")  # URL du modèle Gradio
        result = client.predict(
            prompt,
            512,
            512,
            0,
            20,
            "euler",
            "normal",
            7.5,
            0,
            0,
            fn_index=0,
            timeout=60
        )

        # Chemin temporaire de l'image générée
        image_path = result  
        class_name_clean = clean_filename(class_name.upper())

        class_dir = os.path.join(CLASS1_DIR, class_name_clean)
        os.makedirs(class_dir, exist_ok=True)

        image_name = f"{image_id}.png"
        local_image_path = os.path.join(class_dir, image_name)

        if os.path.exists(image_path):
            with open(image_path, "rb") as src, open(local_image_path, 'wb') as dst:
                dst.write(src.read())
            return {
                "success": True,
                "image_path": local_image_path,
                "message": f"Image {image_name} saved for prompt: {prompt}"
            }
        else:
            raise FileNotFoundError(f"File not found: {image_path}")

    except Exception as e:
        return {
            "success": False,
            "image_path": None,
            "message": str(e)
        }

# Endpoint principal
@app.route('/llm', methods=['POST', 'OPTIONS'])
def process_text():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()
    prompt_text = data.get("prompt", "")
    if not prompt_text:
        return jsonify({"success": False, "message": "Prompt is missing"}), 400

    # Nettoyer ancien dossier class1 s’il existe
    if os.path.exists(CLASS1_DIR):
        shutil.rmtree(CLASS1_DIR)
    os.makedirs(CLASS1_DIR)

    # Utiliser regex pour découper correctement les classes
    class_prompts = re.split(r"\s*,?\s*END\s*,?\s*", prompt_text.strip())
    results = []
    image_id = 64  # Compteur d'ID d'image

    for class_prompt in class_prompts:
        class_prompt = class_prompt.strip()
        if not class_prompt:
            continue
        parts = class_prompt.split("/")
        if len(parts) < 2:
            continue

        class_name = parts[0].strip()
        prompts = [p.strip() for p in parts[1:] if p.strip()]

        for p in prompts:
            result = generate_image_from_prompt(class_name, p, image_id)
            image_id += 1
            results.append(result)

    if not all(r["success"] for r in results):
        return jsonify({
            "success": False,
            "results": results
        }), 500

    # Créer un zip après génération complète
    zip_path = os.path.join(BASE_DIR, "generated_images.zip")
    image_files = []

    for root, _, files in os.walk(CLASS1_DIR):
        for file in files:
            abs_path = os.path.join(root, file)
            image_files.append(abs_path)

    if not image_files:
        return jsonify({"success": False, "message": "No images generated"}), 500

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in image_files:
            zipf.write(file, os.path.relpath(file, BASE_DIR))

    if not os.path.exists(zip_path):
        return jsonify({"success": False, "message": "ZIP file creation failed"}), 500

    return send_file(zip_path, as_attachment=True, download_name="generated_images.zip")

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
