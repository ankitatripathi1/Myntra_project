from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Load & preprocess JSON once
# ---------------------------
with open(os.path.join(BASE_DIR, "video_index.json")) as f:
    RAW_DATA = json.load(f)
    VIDEO_DATA = RAW_DATA.get("videos", [])

# Precompute searchable text for each video
for v in VIDEO_DATA:
    text_parts = v.get("hashtags", [])
    for p in v.get("products", []):
        fields = [p.get("name", ""), p.get("color", ""), p.get("type", ""), p.get("occasion", "")]
        text_parts.extend(fields)
    v["search_text"] = " ".join(text_parts).lower()

# ---------------------------
# Routes
# ---------------------------
@app.route("/api/videos")
def api_videos():
    return jsonify({"videos": VIDEO_DATA})


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.json
    prompt = (data.get("prompt") or "").lower().strip()
    max_price = data.get("max_price")

    if max_price:
        try:
            max_price = float(max_price)
        except:
            max_price = None

    if not prompt and not max_price:
        return jsonify({"videos": VIDEO_DATA})

    prompt_words = prompt.split()

    def score_video(video):
        score = 0

        # match prompt words against precomputed text
        if prompt_words:
            for w in prompt_words:
                if w in video["search_text"]:
                    score += 2

        # apply price filter (check at least one product)
        if max_price:
            has_valid_product = any(
                p.get("price") and p["price"] <= max_price
                for p in video.get("products", [])
            )
            if not has_valid_product:
                return 0  # reject video

        return score

    # compute scores
    scored = [(score_video(v), v) for v in VIDEO_DATA]
    relevant_sorted = [v for s, v in sorted(scored, key=lambda x: x[0], reverse=True) if s > 0]
    irrelevant = [v for s, v in scored if s == 0]
    results = relevant_sorted + irrelevant

    return jsonify({"videos": results})


@app.route("/videos/<path:filename>")
def videos(filename):
    return send_from_directory(os.path.join(BASE_DIR, "videos"), filename)


@app.route("/thumbnails/<path:filename>")
def thumbnails(filename):
    return send_from_directory(os.path.join(BASE_DIR, "thumbnails"), filename)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
