from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, json, base64, io, cv2
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

# ---------------------------
# Setup
# ---------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Color keywords
# ---------------------------
COLORS = [
    "red", "blue", "green", "yellow", "black", "white",
    "pink", "purple", "orange", "brown", "grey", "silver", "gold"
]

# ---------------------------
# Gen Z Slang Dictionary (Expanded)
# ---------------------------
GENZ_SLANG_MAP = {
    "fit": ["outfit", "attire", "ensemble", "look"],
    "drip": ["stylish", "fashionable", "trendy", "cool", "streetwear"],
    "slay": ["stunning", "impressive", "dominant", "amazing"],
    "fire": ["amazing", "great", "excellent", "trendy"],
    "vibe": ["aesthetic", "mood", "feeling", "style"],
    "aesthetic": ["style", "look", "vibe"],
    "boujee": ["luxurious", "expensive", "fancy", "high-end", "glamorous"],
    "comfy": ["comfortable", "cozy", "relaxed"],
    "cheugy": ["outdated", "unfashionable", "uncool"],
    "glow up": ["transformation", "makeover", "improvement"],
    "lewk": ["unique look", "style statement"],
    "clean girl": ["minimalist", "effortless elegance", "chic"],
    "old money": ["classic", "elegant", "preppy", "subtle luxury"],
    "baddie": ["confident", "stylish", "high fashion"],
    "y2k": ["nostalgic", "vintage", "early 2000s"],
    "soft girl": ["cute", "pastel", "cutesy", "girly"],
    "it's giving": ["reminds me of", "gives the feel of", "evokes"],
    "periodt": ["emphasis", "final statement"],
    "bet": ["agreed", "sure"],
    "rizz": ["charisma", "charming"],
    "w": ["win", "success"],
    "l": ["loss", "failure"]
}

def preprocess_prompt(prompt):
    """
    Handles Gen Z slang and expands the query for better search results.
    """
    prompt = prompt.lower().strip()
    words = prompt.split()
    expanded_words = list(words)
    
    for word in words:
        if word in GENZ_SLANG_MAP:
            expanded_words.extend(GENZ_SLANG_MAP[word])
    
    return " ".join(sorted(list(set(expanded_words))))

# ---------------------------
# Load dataset
# ---------------------------
with open(os.path.join(BASE_DIR, "video_index.json")) as f:
    RAW_DATA = json.load(f)
    VIDEO_DATA = RAW_DATA.get("videos", [])

def build_product_text(p):
    return f"{p.get('color','')} {p.get('type','')} {p.get('name','')} {p.get('occasion','')}".strip().lower()

for v in VIDEO_DATA:
    text_parts = v.get("hashtags", [])
    for p in v.get("products", []):
        text_parts.append(build_product_text(p))
    v["search_text"] = " ".join(text_parts).lower()

# ---------------------------
# Load CLIP model
# ---------------------------
print("🔄 Loading CLIP model...")
embedder = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
print("✅ Model loaded!")

# ---------------------------
# Embedding helpers
# ---------------------------
def encode_text(text):
    return embedder.encode(text, convert_to_tensor=True)

def encode_image(image_b64):
    try:
        image_data = base64.b64decode(image_b64.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return embedder.encode(image, convert_to_tensor=True)
    except Exception as e:
        print("⚠️ Image embedding failed:", e)
        return None

def extract_video_embedding(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)
    frames = []

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        emb = embedder.encode(image, convert_to_tensor=True)
        frames.append(emb)

    cap.release()
    if not frames:
        return None
    return torch.mean(torch.stack(frames), dim=0)

# ---------------------------
# Precompute embeddings
# ---------------------------
print("🔄 Precomputing embeddings...")
for v in VIDEO_DATA:
    v["text_embedding"] = encode_text(v["search_text"])
    video_path = os.path.join(BASE_DIR, v["video_url"].lstrip("/"))
    v["visual_embedding"] = extract_video_embedding(video_path) if os.path.exists(video_path) else None

    for p in v.get("products", []):
        product_text = build_product_text(p)
        if product_text:
            p["embedding"] = encode_text(product_text)
        else:
            p["embedding"] = None
print("✅ Embeddings ready!")

# ---------------------------
# Color bonus
# ---------------------------
def color_match_bonus(prompt, product):
    score = 0
    for c in COLORS:
        if c in prompt and product.get("color","").lower() == c:
            score += 5
            print(f"🎨 Color match bonus: '{c}' found in query and product {product.get('name')}")
    return score

# ---------------------------
# Scoring logic (IMPROVED)
# ---------------------------
def score_video_improved(video, query_embedding, image_embedding, prompt, max_price):
    
    # Use .get() to safely handle missing 'video_id' or 'id' keys.
    video_id = video.get('video_id') or video.get('id', 'N/A')

    # 0. Price Filter
    if max_price:
        has_valid_product = any(
            p.get("price") and p["price"] <= max_price for p in video.get("products", [])
        )
        if not has_valid_product:
            print(f"💰 Video {video_id} filtered out due to price.")
            return -999

    text_score = 0
    image_score = 0

    # 1. Calculate Text-based Score
    if query_embedding is not None:
        max_product_text_sim = 0
        for p in video.get("products", []):
            if p.get("embedding") is not None:
                sim = torch.cosine_similarity(query_embedding, p["embedding"], dim=0).item()
                max_product_text_sim = max(max_product_text_sim, sim)

        visual_text_sim = 0
        if video.get("visual_embedding") is not None:
            visual_text_sim = torch.cosine_similarity(query_embedding, video["visual_embedding"], dim=0).item()
        
        text_score = (max_product_text_sim * 0.6) + (visual_text_sim * 0.4)
        print(f"📝 Text Score for {video_id}: Product Sim={max_product_text_sim:.3f}, Visual Sim={visual_text_sim:.3f}, Final={text_score:.3f}")

        prompt_words = set(prompt.split())
        search_text_words = video["search_text"].split()
        
        for p_word in prompt_words:
            for s_word in search_text_words:
                if fuzz.ratio(p_word, s_word) > 85:
                    text_score += 0.2
                    break

            for p in video.get("products", []):
                text_score += color_match_bonus(prompt, p)

    # 2. Calculate Image-based Score
    if image_embedding is not None:
        max_product_image_sim = 0
        for p in video.get("products", []):
            if p.get("embedding") is not None:
                sim = torch.cosine_similarity(image_embedding, p["embedding"], dim=0).item()
                max_product_image_sim = max(max_product_image_sim, sim)

        visual_image_sim = 0
        if video.get("visual_embedding") is not None:
            visual_image_sim = torch.cosine_similarity(image_embedding, video["visual_embedding"], dim=0).item()
        
        image_score = (max_product_image_sim * 0.6) + (visual_image_sim * 0.4)
        print(f"🖼️ Image Score for {video_id}: Product Sim={max_product_image_sim:.3f}, Visual Sim={visual_image_sim:.3f}, Final={image_score:.3f}")

    # 3. Final Fusion Score
    final_score = 0
    if query_embedding is not None and image_embedding is not None:
        final_score = (text_score + image_score) / 2
        if text_score > 0.5 and image_score > 0.5:
             final_score += (text_score + image_score) * 0.5
    elif query_embedding is not None:
        final_score = text_score
    elif image_embedding is not None:
        final_score = image_score
    else:
        return 0

    return final_score

# ---------------------------
# JSON Cleaner
# ---------------------------
def clean_for_json(video):
    """Remove or convert non-serializable fields like torch.Tensors."""
    clean_video = {}
    for k, v in video.items():
        if k in ["text_embedding", "visual_embedding"]:
            continue
        if isinstance(v, torch.Tensor):
            continue
        elif isinstance(v, list):
            clean_video[k] = [clean_for_json(x) if isinstance(x, dict) else x for x in v]
        elif isinstance(v, dict):
            clean_video[k] = clean_for_json(v)
        else:
            clean_video[k] = v
    return clean_video

# ---------------------------
# Routes
# ---------------------------
@app.route("/api/videos")
def api_videos():
    safe_data = [clean_for_json(v) for v in VIDEO_DATA]
    return jsonify({"videos": safe_data})

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.json
    raw_prompt = data.get("prompt", "")
    max_price = data.get("max_price")
    image_b64 = data.get("image")

    prompt = preprocess_prompt(raw_prompt) 

    if max_price:
        try:
            max_price = float(max_price)
        except:
            max_price = None

    query_embedding = encode_text(prompt) if prompt else None
    image_embedding = encode_image(image_b64) if image_b64 else None

    scored = [(score_video_improved(v, query_embedding, image_embedding, raw_prompt.lower().strip(), max_price), v) for v in VIDEO_DATA]
    sorted_videos = [clean_for_json(v) 
                     for s, v in sorted(scored, key=lambda x: x[0], reverse=True) if s > -999]

    return jsonify({"videos": sorted_videos})

@app.route("/videos/<path:filename>")
def videos(filename):
    return send_from_directory(os.path.join(BASE_DIR, "videos"), filename)

@app.route("/thumbnails/<path:filename>")
def thumbnails(filename):
    return send_from_directory(os.path.join(BASE_DIR, "thumbnails"), filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
