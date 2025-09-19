import os
import json
from flask import Flask, render_template, request, jsonify, url_for
from clip_image_similarity import CLIPEmbedder

# Initialize the Flask app
app = Flask(__name__)

print("Initializing CLIP model...")
embedder = CLIPEmbedder()
print("CLIP model loaded.")

# --- Helper Function ---
def get_image_data():
    """Scans for images and loads metadata."""
    image_paths = []
    base_path = "static/screenshots"
    for folder in ["cropped", "fullscreen"]:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(f"screenshots/{folder}/{filename}")

    # Load metadata
    with open("image_metadata.json", "r") as f:
        metadata = json.load(f)
        
    return {"image_paths": sorted(image_paths), "metadata": metadata}

# --- API Routes ---

@app.route("/")
def index():
    """Renders the main HTML page."""
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    """Provides the initial data (image paths and metadata) to the frontend."""
    return jsonify(get_image_data())


@app.route("/api/similarity", methods=["POST"])
def api_similarity():
    """Calculates and returns the similarity between two images."""
    data = request.get_json()
    
    # Get the two image paths from the request
    img1_relative_path = data.get("image1_path")
    img2_relative_path = data.get("image2_path")

    if not img1_relative_path or not img2_relative_path:
        return jsonify({"error": "Two images must be selected."}), 400

    # Construct the full file system path for the model
    img1_full_path = os.path.join("static", img1_relative_path)
    img2_full_path = os.path.join("static", img2_relative_path)

    # Embed each image
    emb1 = embedder.embed_image(img1_full_path)
    emb2 = embedder.embed_image(img2_full_path)
    
    # Calculate similarity
    similarity_score = embedder.calculate_similarity(emb1, emb2)

    return jsonify({"similarity": similarity_score})

if __name__ == "__main__":
    app.run(debug=True)