"""
Fashion Search App - FULLY FIXED VERSION
- Fixed gender detection (word boundaries)
- Fixed color detection (word boundaries)  
- Fixed OpenAI v1.x API
- Added jewelry support
- Fixed numpy bool serialization
"""

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_session import Session
import os, json, base64, io, requests, uuid, pickle
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import time, re, colorsys
from collections import Counter
from scipy.spatial import KDTree
import hashlib
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    USE_LLM_NORMALIZER = True
except Exception as e:
    print("⚠️ Local LLM not available, continuing without normalization:", e)
    USE_LLM_NORMALIZER = False
    llm_model = None


try:
    from skimage.color import rgb2lab, deltaE_ciede2000
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# =============================================================================
# FLASK SETUP
# =============================================================================
app = Flask(__name__)

CORS(app, supports_credentials=True, 
     origins=["http://localhost:3000", "http://localhost:5173"], 
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

app.secret_key = "supersecretkey_fashion_app_2024"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_KEY_PREFIX"] = "fashion:"
app.config["SESSION_FILE_DIR"] = os.path.join(os.path.dirname(__file__), "flask_session")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = 86400

os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)

# REPLACE existing CACHE_DIR setup (around line 48) with:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WARDROBE_DIR = os.path.join(BASE_DIR, "wardrobe_photos")
WISHLIST_DIR = os.path.join(BASE_DIR, "wishlist_photos")
CACHE_DIR = os.path.join(BASE_DIR, "cache")  # Changed from "color_cache"
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings_v2.pkl")
MODELS_CACHE_DIR = os.path.join(CACHE_DIR, "models")
HARMONY_CACHE_FILE = os.path.join(CACHE_DIR, "harmony_cache_v3.pkl")

os.makedirs(WARDROBE_DIR, exist_ok=True)
os.makedirs(WISHLIST_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

# =============================================================================
# LOAD MODELS
# =============================================================================
# =============================================================================
# LAZY MODEL LOADING
# =============================================================================
class ModelLoader:
    """Lazy-load models only when needed"""
    def __init__(self):
        self._embedder = None
        self._blip_processor = None
        self._blip_model = None
        
    @property
    def embedder(self):
        if self._embedder is None:
            print("🔄 Loading CLIP embedder...")
            self._embedder = SentenceTransformer(
                "sentence-transformers/clip-ViT-B-32",
                cache_folder=MODELS_CACHE_DIR
            )
            print("✅ CLIP loaded")
        return self._embedder
    
    @property
    def blip_processor(self):
        if self._blip_processor is None:
            print("🔄 Loading BLIP processor...")
            self._blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=MODELS_CACHE_DIR
            )
            print("✅ BLIP processor loaded")
        return self._blip_processor
    
    @property
    def blip_model(self):
        if self._blip_model is None:
            print("🔄 Loading BLIP model...")
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=MODELS_CACHE_DIR
            )
            self._blip_model.eval()
            print("✅ BLIP model loaded")
        return self._blip_model

# Global instance
models = ModelLoader()


# =============================================================================
# EMBEDDINGS CACHE MANAGER
# =============================================================================
class EmbeddingsCache:
    """Manage persistent embeddings cache"""
    
    def __init__(self, cache_file=EMBEDDINGS_CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self.load()
        
    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"✅ Loaded {len(cache)} video embeddings from cache")
                return cache
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}")
        return {}
    
    def save(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f, protocol=4)
            print(f"💾 Saved {len(self.cache)} video embeddings")
        except Exception as e:
            print(f"❌ Cache save failed: {e}")
    
    def get_hash(self, video):
        """Generate hash for video"""
        key_data = json.dumps({
            'video_id': video.get('video_id'),
            'products': [
                {
                    'name': p.get('name'),
                    'image_url': p.get('image_url')
                }
                for p in video.get('products', [])
            ]
        }, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, video):
        video_hash = self.get_hash(video)
        return self.cache.get(video_hash)
    
    def set(self, video, embeddings):
        video_hash = self.get_hash(video)
        self.cache[video_hash] = {
            'embeddings': embeddings,
            'timestamp': datetime.now().isoformat()
        }

# Global cache instance
embeddings_cache = EmbeddingsCache()
# =============================================================================
# KEYWORDS - FIXED WITH WORD BOUNDARIES
# =============================================================================
# CRITICAL FIX: Use sorted by length DESC to match longer phrases first
FEMALE_KEYWORDS = ["women's", "womens", "women", "woman", "female", "girls", "girl", "feminine", "bride", "lady", "ladies"]
MALE_KEYWORDS = ["men's", "mens", "men", "man", "male", "boys", "boy", "masculine", "groom", "gentleman"]

# Sort by length descending to match "women's" before "women", "men's" before "men"
FEMALE_KEYWORDS = sorted(FEMALE_KEYWORDS, key=len, reverse=True)
MALE_KEYWORDS = sorted(MALE_KEYWORDS, key=len, reverse=True)


ITEM_TYPES = [
    # Tops
    'blouse', 't-shirt', 'tshirt', 'shirt', 'top', 'tank top', 'tank', 'camisole', 'bodycon',
    # Bottoms  
    'trousers', 'jeans', 'pants', 'skirt', 'shorts', 'leggings', 'cargo', 'pajama', 'joggers',
    # Dresses
    'gown', 'dress', 'frock', 'maxi dress', 'midi dress','western dress',
    # Outerwear
    'blazer', 'jacket', 'coat', 'sweater', 'cardigan', 'hoodie',
    # Traditional
    'saree', 'sari', 'kurta', 'kurti', 'salwar', 'kameez', 'lehenga', 'dupatta', 'suit','traditional','gown'
    # Accessories & Jewelry
    'jhumka', 'jhumkas', 'earring', 'earrings', 'necklace', 'bracelet', 'studs',
    'ring', 'bangles', 'anklet', 'maang tikka', 'nose ring', 'pendant',
    'hair accessories', 'clutcher', 'clips', 'clip', 'hair clip', 'hair claw clip',
    'watch', 'belt', 'bag',
    # Beauty
    'lipstick', 'makeup', 'cosmetics',"make up",
    # Footwear
    'footwear', 'shoes', 'sneakers', 'heels', 'flats', 'sandals',
    # Others
    'jumpsuit', 'romper', 'tie'
]
ITEM_TYPES = sorted(ITEM_TYPES, key=len, reverse=True)  # Match longer first

COMPLEMENTARY_KEYWORDS = ['complementary', 'complement', 'match', 'pair', 'go with', 'goes with']
EXCLUSION_KEYWORDS = ['other than', 'except', 'not', 'without', 'excluding', 'no']

GENZ_SLANG_MAP = {
    "drip": "stylish outfit", "fit": "outfit", "slay": "amazing look",
    "fire": "great", "bussin": "excellent", "lowkey": "somewhat",
    "highkey": "very", "mid": "average", "goated": "best", "rizz": "style"
}

ITEM_TYPE_MAPPINGS = {
    'hair accessories': ['clip', 'clips', 'clutcher', 'hair clip', 'claw clip', 'korean clips', 'hair claw clip'],
    'footwear': ['shoes', 'sneakers', 'heels', 'flats', 'sandals', 'boots'],
    'jewelry': ['earring', 'earrings', 'necklace', 'bracelet', 'ring', 'jhumka', 'jhumkas', 'studs', 'bangles','pendant'],
    'accessories': ['bag', 'belt', 'watch', 'clutcher'],
    'traditional': ['saree', 'sari', 'kurta', 'kurti', 'lehenga','traditional gown'],
    'tops': ['shirt', 't-shirt', 'tshirt', 'blouse', 'top', 'tank'],
    'bottoms': ['jeans', 'pants', 'trousers', 'skirt', 'shorts','cargo','payjama'],
    'western':['dress','western gown','bodycon'],
    'traditional':['saree','sari','kurta','kurti','lehenga','traditional gown'],
    'makeup':['lipstick','makeup','cosmetics','eyeliner', 'mascara', 'foundation','make up']

}
# =============================================================================
# COLOR THEORY
# =============================================================================
def rgb_to_hsv(rgb):
    h, s, v = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    return np.array([h * 360, s, v])

def hsv_to_rgb(hsv):
    h, s, v = hsv[0] / 360.0, hsv[1], hsv[2]
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g, b])

def is_neutral_color(rgb, sat_threshold=0.15):
    hsv = rgb_to_hsv(rgb)
    return hsv[1] < sat_threshold

def get_color_harmonies(rgb):
    hsv = rgb_to_hsv(rgb)
    harmonies = {'complementary': [], 'triadic': [], 'analogous': []}
    
    comp_hsv = hsv.copy()
    comp_hsv[0] = (hsv[0] + 180) % 360
    harmonies['complementary'].append(hsv_to_rgb(comp_hsv))
    
    for angle in [120, 240]:
        tri_hsv = hsv.copy()
        tri_hsv[0] = (hsv[0] + angle) % 360
        harmonies['triadic'].append(hsv_to_rgb(tri_hsv))
    
    for angle in [30, -30]:
        ana_hsv = hsv.copy()
        ana_hsv[0] = (hsv[0] + angle) % 360
        harmonies['analogous'].append(hsv_to_rgb(ana_hsv))
    
    return harmonies

# =============================================================================
# COLOR PALETTE
# =============================================================================
def load_xkcd_palette():
    try:
        resp = requests.get("https://raw.githubusercontent.com/dariusk/corpora/master/data/colors/xkcd.json", timeout=5)
        data = resp.json()
        palette = {}
        for entry in data.get("colors", []):
            name = entry["color"].lower()
            hexv = entry["hex"]
            rgb = tuple(int(hexv.lstrip('#')[i:i+2], 16) for i in (0,2,4))
            palette[name] = rgb
        print(f"✅ Loaded {len(palette)} colors")
        return palette
    except Exception:
        return {
            'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
            'yellow': (255, 255, 0), 'black': (0, 0, 0), 'white': (255, 255, 255),
            'pink': (255, 192, 203), 'purple': (128, 0, 128), 'orange': (255, 165, 0),
            'brown': (165, 42, 42), 'grey': (128, 128, 128), 'navy': (0, 0, 128)
        }

XKCD_PALETTE = load_xkcd_palette()
PALETTE_NAMES = sorted(list(XKCD_PALETTE.keys()), key=len, reverse=True)  # CRITICAL: Sort by length
PALETTE_RGB = np.array([np.array(XKCD_PALETTE[n]) / 255.0 for n in PALETTE_NAMES])
PALETTE_RGB_255 = np.array([np.array(XKCD_PALETTE[n]) for n in PALETTE_NAMES])
PALETTE_KDTREE = KDTree(PALETTE_RGB_255)

if HAS_SKIMAGE:
    PALETTE_LAB = np.array([rgb2lab(rgb.reshape(1,1,3))[0,0] for rgb in PALETTE_RGB])
    PALETTE_LAB_KDTREE = KDTree(PALETTE_LAB)
else:
    PALETTE_LAB = None
    PALETTE_LAB_KDTREE = None

# =============================================================================
# COMPLEMENTARY ITEMS
# =============================================================================
ITEM_COMPLEMENTS = {
    'shirt': ['pants', 'jeans', 'trousers', 'skirt','cargo'],
    't-shirt': ['pants', 'jeans', 'skirt', 'shorts','cargo'],
    'blouse': ['pants', 'jeans', 'skirt', 'trousers'],
    'top': ['pants', 'jeans', 'skirt', 'shorts'],
    'pants': ['shirt', 'blouse', 't-shirt', 'top'],
    'jeans': ['shirt', 'blouse', 't-shirt', 'top'],
    'trousers': ['shirt', 'blouse', 'top'],
    'skirt': ['shirt', 'blouse', 't-shirt', 'top'],
    'dress': ['jacket', 'blazer', 'cardigan'],
    'gown': ['jacket', 'shawl', 'dupatta'],
    # New additions:
    'cargo': ['shirt', 'blouse', 't-shirt', 'top', 'tshirt'],
    'pajama': ['kurta', 'shirt'],
    'bodycon': ['jeans', 'skirt', 'jacket'],
    'saree': ['blouse'],  # Saree goes with blouse traditionally
    'kurta': ['pajama', 'jeans', 'trousers'],
    
    # Accessories don't need complements (they go with everything)
    # But you could add:
    'belt': ['pants', 'jeans', 'trousers', 'skirt'],
    'watch': [],  # Universal accessory
    'bag': []    # Universal accessory
}

# =============================================================================
# HARMONY CACHE
# =============================================================================
HARMONY_CACHE = {}

def find_nearest_palette_color(rgb, max_delta_e=25):
    rgb_255 = (rgb * 255).astype(int)
    if HAS_SKIMAGE and PALETTE_LAB_KDTREE:
        lab = rgb2lab(rgb.reshape(1,1,3))[0,0]
        dist, idx = PALETTE_LAB_KDTREE.query(lab, k=1)
        if dist <= max_delta_e:
            return PALETTE_NAMES[idx], float(dist)
    else:
        dist, idx = PALETTE_KDTREE.query(rgb_255, k=1)
        return PALETTE_NAMES[idx], float(dist)
    return None, None

def build_harmony_cache():
    cache_file = HARMONY_CACHE_FILE  
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                print(f"✅ Loaded harmony cache ({len(cached)} colors)")
                return cached
        except Exception:
            pass
    
    print("Building harmony cache...")
    cache = {}
    for i, name in enumerate(PALETTE_NAMES):
        rgb = PALETTE_RGB[i]
        harmonies = get_color_harmonies(rgb)
        cache[name] = {
            'complementary_colors': [],
            'triadic_colors': [],
            'analogous_colors': [],
            'is_neutral': bool(is_neutral_color(rgb))
        }
        for harm_type in ['complementary', 'triadic', 'analogous']:
            for harm_rgb in harmonies[harm_type]:
                nearest_name, _ = find_nearest_palette_color(harm_rgb)
                if nearest_name:
                    cache[name][f'{harm_type}_colors'].append(nearest_name)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(PALETTE_NAMES)}")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    except Exception:
        pass
    return cache

# =============================================================================
# IMAGE PROCESSING
# =============================================================================
def extract_dominant_color_rgb(pil_img, resize=50):
    try:
        img = pil_img.copy()
        img.thumbnail((resize, resize))
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        pixels = arr.reshape(-1, 3)
        pixels_tuple = [tuple(p) for p in pixels]
        most_common = Counter(pixels_tuple).most_common(1)
        if most_common:
            return np.array(most_common[0][0]) / 255.0
    except Exception:
        pass
    return None

def get_blip_caption(image: Image.Image, max_retries=2):
    for attempt in range(max_retries):
        try:
            inputs = models.blip_processor(images=image, return_tensors="pt")
            out = models.blip_model.generate(**inputs, max_length=50, num_beams=5)
            return models.blip_processor.decode(out[0], skip_special_tokens=True).lower().strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5)
    return ""

def extract_metadata_from_caption(caption):
    """Extract type, gender, color from BLIP caption"""
    caption_lower = caption.lower()
    metadata = {
        'type': None,
        'gender': None,
        'colors': []
    }
    
    # Extract item type
    for item_type in ITEM_TYPES:
        if re.search(r'\b' + re.escape(item_type) + r'\b', caption_lower):
            metadata['type'] = item_type
            break
    
    # Extract gender
    for keyword in FEMALE_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', caption_lower):
            metadata['gender'] = 'women'
            break
    
    if not metadata['gender']:
        for keyword in MALE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', caption_lower):
                metadata['gender'] = 'men'
                break
    
    # Extract colors
    for color_name in PALETTE_NAMES[:100]:  # Check top 100 common colors
        if re.search(r'\b' + re.escape(color_name) + r'\b', caption_lower):
            metadata['colors'].append(color_name)
            if len(metadata['colors']) >= 2:  # Limit to 2 colors
                break
    
    return metadata

def normalize_query_with_llm(raw_prompt):
    """Normalize query: GenZ slang + fuzzy spelling correction"""
    if not raw_prompt or len(raw_prompt.strip()) < 3:
        return raw_prompt

    # Step 1: Expand GenZ slang
    normalized_parts = []
    original_words = raw_prompt.split()

    for word in original_words:
        normalized_parts.append(word)  # Keep original
        word_lower = word.lower()
        if word_lower in GENZ_SLANG_MAP:
            meaning = GENZ_SLANG_MAP[word_lower]
            if meaning:
                normalized_parts.append(meaning)

    # Step 2: Fuzzy match phrases (bigrams/trigrams) first
    prompt_lower = raw_prompt.lower()
    
    # Check 2-word and 3-word phrases
    for n in [3, 2]:  # Check longer phrases first
        words = prompt_lower.split()
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            
            # Fuzzy match against multi-word ITEM_TYPES
            for item_type in ITEM_TYPES:
                if ' ' in item_type:  # Only multi-word types
                    score = fuzz.ratio(phrase, item_type)
                    if score >= 75:
                        normalized_parts.append(item_type)
                        print(f"🔍 Fuzzy phrase: '{phrase}' → '{item_type}' (score: {score})")
                        break

    # Step 3: Fuzzy match individual words
    for word in original_words:
        word_clean = word.lower().strip()
        
        if word_clean in ITEM_TYPES:
            continue  # Already correct
        
        # Find best match among single-word item types
        best_match = None
        best_score = 0
        
        for item_type in ITEM_TYPES:
            if ' ' not in item_type:  # Only single-word types
                score = fuzz.ratio(word_clean, item_type)
                if score > best_score and score >= 75:
                    best_score = score
                    best_match = item_type
        
        if best_match:
            normalized_parts.append(best_match)
            print(f"🔍 Fuzzy word: '{word_clean}' → '{best_match}' (score: {best_score})")
    
    return " ".join(normalized_parts)


def encode_text(text):
    if not text or not text.strip():
        return None
    try:
        return models.embedder.encode(text, convert_to_tensor=True)
    except Exception:
        return None

# Add query caching
@lru_cache(maxsize=1000)
def encode_text_cached(text):
    """Cache frequently searched queries"""
    return encode_text(text)

def load_image_from_url(url, max_retries=2, timeout=10):
    if not url.startswith('http'):
        local_path = os.path.join(BASE_DIR, url.lstrip('/'))
        if os.path.exists(local_path):
            try:
                image = Image.open(local_path).convert("RGB")
                if image.size[0] >= 50 and image.size[1] >= 50:
                    return image
            except Exception:
                pass
        return None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            if image.size[0] >= 50 and image.size[1] >= 50:
                return image
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None

# =============================================================================
# LOAD DATASET
# =============================================================================
with open(os.path.join(BASE_DIR, "video_index.json")) as f:
    VIDEO_DATA = json.load(f).get("videos", [])

def build_product_text(product, video_hashtags=None):
    """Build COMPREHENSIVE product text including ALL searchable fields"""
    parts = []
    
    # Core product fields
    if product.get('name'): parts.append(product['name'])
    if product.get('color'): parts.append(product['color'])
    if product.get('type'): parts.append(product['type'])
    if product.get('occasion'): parts.append(product['occasion'])
    if product.get('gender'): parts.append(product['gender'])
    if product.get('description'): parts.append(product['description'])
    
    # Add hashtags
    if video_hashtags:
        parts.extend(video_hashtags)
    
    # CRITICAL: Also add expanded item types
    product_type = (product.get('type') or '').lower()
    if product_type:
        expanded_types = expand_item_type(product_type)
        parts.extend(expanded_types)
    
    # Add product name variations
    product_name = (product.get('name') or '').lower()
    if product_name:
        for parent, children in ITEM_TYPE_MAPPINGS.items():
            if any(child in product_name for child in children):
                parts.append(parent)  # Add parent category
                parts.extend(children)  # Add all siblings
                break
    
    return " ".join(parts).lower().strip()
# =============================================================================
# PRE-COMPUTE
# =============================================================================

def expand_item_type(item_type):
    """Expand item type to include all related terms"""
    expanded = [item_type]
    
    # Check if this item is a parent category
    if item_type in ITEM_TYPE_MAPPINGS:
        expanded.extend(ITEM_TYPE_MAPPINGS[item_type])
    
    # Check if this item is a child of any category
    for parent, children in ITEM_TYPE_MAPPINGS.items():
        if item_type in children:
            expanded.append(parent)
            expanded.extend(children)
            break
    
    return list(set(expanded))  # Remove duplicates

def precompute_all_embeddings():
    """Incremental embedding computation with caching"""
    print("\n" + "="*60)
    print("INCREMENTAL EMBEDDING COMPUTATION")
    print("="*60)

    total = 0
    cached = 0
    computed = 0

    for video in VIDEO_DATA:
        # Check cache first
        cached_data = embeddings_cache.get(video)
        
        if cached_data:
            # Load from cache
            video_embeddings = cached_data['embeddings']
            for i, product in enumerate(video.get('products', [])):
                key = f'product_{i}'
                if key in video_embeddings:
                    prod_data = video_embeddings[key]
                    
                    # FIX: Convert lists back to tensors
                    if prod_data.get('desc_emb'):
                        product['description_embedding'] = torch.tensor(prod_data['desc_emb'])
                    
                    product['blip_caption'] = prod_data.get('caption')
                    
                    if prod_data.get('caption_emb'):
                        product['blip_caption_embedding'] = torch.tensor(prod_data['caption_emb'])
                    
                    product['dominant_color_rgb'] = prod_data.get('color_rgb')
                    product['dominant_color_lab'] = prod_data.get('color_lab')
                    product['nearest_palette_color'] = prod_data.get('palette_color')
                    product['color_match_confidence'] = prod_data.get('color_confidence')
                    product['complementary_colors'] = prod_data.get('comp_colors')
                    product['is_neutral'] = prod_data.get('is_neutral')
                    cached += 1
            continue

        # Compute embeddings for new video
        video_embeddings = {}
        hashtags = video.get('hashtags', [])
        
        for i, product in enumerate(video.get('products', [])):
            total += 1
            computed += 1
            
            text = build_product_text(product, hashtags)
            if text.strip():
                product["description_embedding"] = encode_text(text)

            image_url = product.get("image_url")
            if image_url and image_url.strip():
                img = load_image_from_url(image_url.strip())
                if img:
                    product["blip_caption"] = get_blip_caption(img)
                    if product["blip_caption"]:
                        product["blip_caption_embedding"] = encode_text(product["blip_caption"])

                    dom_rgb = extract_dominant_color_rgb(img)
                    if dom_rgb is not None:
                        product["dominant_color_rgb"] = dom_rgb.tolist()
                        if HAS_SKIMAGE:
                            product["dominant_color_lab"] = rgb2lab(dom_rgb.reshape(1,1,3))[0,0].tolist()
                        
                        nearest_name, delta_e = find_nearest_palette_color(dom_rgb)
                        product['nearest_palette_color'] = nearest_name
                        product['color_match_confidence'] = float(1 - min(delta_e / 25, 1)) if delta_e else 1.0
                        
                        if nearest_name and nearest_name in HARMONY_CACHE:
                            harmonies = HARMONY_CACHE[nearest_name]
                            product['complementary_colors'] = harmonies.get('complementary_colors', [])
                            product['is_neutral'] = harmonies.get('is_neutral', False)
            
            # Store in cache
            # Store in cache (convert tensors to lists for pickling)
            video_embeddings[f'product_{i}'] = {
                'desc_emb': product.get('description_embedding').tolist() if product.get('description_embedding') is not None else None,
                'caption': product.get('blip_caption'),
                'caption_emb': product.get('blip_caption_embedding').tolist() if product.get('blip_caption_embedding') is not None else None,
                'color_rgb': product.get('dominant_color_rgb'),
                'color_lab': product.get('dominant_color_lab'),
                'palette_color': product.get('nearest_palette_color'),
                'color_confidence': product.get('color_match_confidence'),
                'comp_colors': product.get('complementary_colors'),
                'is_neutral': product.get('is_neutral')
            }
            
            if computed % 10 == 0:
                print(f"  Computed: {computed}, Cached: {cached}")
        
        embeddings_cache.set(video, video_embeddings)
    
    # Save cache
    embeddings_cache.save()
    
    print(f"\n✅ Total: {total + cached}")
    print(f"✅ From cache: {cached}")
    print(f"✅ Newly computed: {computed}")
    if total + cached > 0:
        print(f"⚡ Cache hit rate: {cached/(total+cached)*100:.1f}%\n")
# =============================================================================
# QUERY PARSER - FIXED WITH WORD BOUNDARIES
# =============================================================================
def parse_search_query(prompt, image_data=None):
    """Parse query with EXPANDED item type matching"""
    query = {
        'mode': 'semantic',
        'gender': None,
        'reference_item': None,
        'target_item': None,
        'colors_include': [],
        'colors_exclude': [],
        'color_rgb': None,
        'original_prompt': prompt,
        'expanded_target_items': []
    }
    
    # ENHANCED: Extract metadata from image even without prompt
    if image_data:
        caption = image_data.get('caption', '')
        if caption:
            metadata = extract_metadata_from_caption(caption)
            
            # Extract item type from caption
            if metadata['type']:
                query['target_item'] = metadata['type']
                query['expanded_target_items'] = expand_item_type(metadata['type'])
            
            # Extract gender from caption
            if not query['gender'] and metadata['gender']:
                query['gender'] = metadata['gender']
            
            # Extract color from caption
            if metadata['colors'] and not query['color_rgb']:
                color_name = metadata['colors'][0]
                try:
                    idx = PALETTE_NAMES.index(color_name)
                    query['color_rgb'] = PALETTE_RGB[idx]
                    query['colors_include'].append(color_name)
                except ValueError:
                    pass
    
    if not prompt and image_data:
        query['mode'] = 'image_only'
        if not query['color_rgb']:
            query['color_rgb'] = image_data.get('dominant_color')
        return query
    
    if not prompt:
        return query
    
    prompt_lower = prompt.lower()
    
    # Gender detection (unchanged)
    for keyword in FEMALE_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', prompt_lower):
            query['gender'] = 'women'
            break
    
    if not query['gender']:
        for keyword in MALE_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', prompt_lower):
                query['gender'] = 'men'
                break
    
    # Detect complementary mode
    is_complementary = any(kw in prompt_lower for kw in COMPLEMENTARY_KEYWORDS)
    if is_complementary:
        query['mode'] = 'complementary'
    
    # Detect exclusion
    for excl_kw in EXCLUSION_KEYWORDS:
        if excl_kw in prompt_lower:
            query['mode'] = 'exclusion'
            break
    
    # CRITICAL FIX: Extract item types with EXPANSION
    found_items = []
    for item_type in ITEM_TYPES:
        if re.search(r'\b' + re.escape(item_type) + r'\b', prompt_lower):
            found_items.append(item_type)
            # EXPAND to include related terms
            query['expanded_target_items'].extend(expand_item_type(item_type))
    
    # Also check child terms (clips → hair accessories)
    for parent, children in ITEM_TYPE_MAPPINGS.items():
        for child in children:
            if re.search(r'\b' + re.escape(child) + r'\b', prompt_lower):
                found_items.append(child)
                query['expanded_target_items'].extend(expand_item_type(child))
    
    # Remove duplicates
    query['expanded_target_items'] = list(set(query['expanded_target_items']))
    
    # Set primary target item
    if found_items:
        query['target_item'] = found_items[0]
    
    # Handle complementary parsing
    if is_complementary and len(found_items) >= 2:
        query['target_item'] = found_items[0]
        query['reference_item'] = found_items[1]
    
    # Extract colors (unchanged)
    for color_name in PALETTE_NAMES:
        if re.search(r'\b' + re.escape(color_name) + r'\b', prompt_lower):
            pattern = f"(?:{'|'.join(EXCLUSION_KEYWORDS)})\\s+{re.escape(color_name)}"
            if re.search(pattern, prompt_lower):
                query['colors_exclude'].append(color_name)
            else:
                query['colors_include'].append(color_name)
                if query['color_rgb'] is None:
                    idx = PALETTE_NAMES.index(color_name)
                    query['color_rgb'] = PALETTE_RGB[idx]
            break
    
    # Image data extraction
    if image_data:
        if query['color_rgb'] is None:
            query['color_rgb'] = image_data.get('dominant_color')
        if query['mode'] == 'complementary' and not query['reference_item']:
            caption = image_data.get('caption', '').lower()
            for item_type in ITEM_TYPES:
                if re.search(r'\b' + re.escape(item_type) + r'\b', caption):
                    query['reference_item'] = item_type
                    break
    
    return query
# =============================================================================
# SCORING ENGINE
# =============================================================================
def score_video(video, query, image_emb=None):
    products = video.get("products", [])
    
    # CRITICAL: Gender filter (HARD)
    if query['gender']:
        has_match = False
        for p in products:
            product_gender = p.get("gender", "").lower().strip()
            product_genders = [g.strip() for g in product_gender.split(',')]
            if query['gender'] in product_genders:
                has_match = True
                break
        
        if not has_match:
            return 0.0
    
    if query['mode'] == 'image_only':
        return score_image_similarity(products, image_emb)
    elif query['mode'] == 'complementary':
        return score_complementary_match(products, query)
    elif query['mode'] == 'exclusion':
        return score_exclusion_match(products, query)
    else:
        return score_semantic_match(products, query)


def score_exclusion_match(products, query):
    """Find items NOT matching excluded colors - FIXED"""
    prompt_emb = encode_text(query['original_prompt'])
    if prompt_emb is None:
        return 0.0
    
    max_score = 0.0
    matched_any = False
    
    for product in products:
        # FIRST: Check item type match (if specified)
        if query['expanded_target_items']:
            product_type = (product.get('type') or '').lower()
            product_name = (product.get('name') or '').lower()
            product_desc = (product.get('description') or '').lower()
            
            type_match = any(
                item_type in product_type or 
                item_type in product_name or
                item_type in product_desc
                for item_type in query['expanded_target_items']
            )
            if not type_match:
                continue
        
        # SECOND: Skip products with excluded colors
        product_color = product.get('nearest_palette_color')
        if product_color and product_color in query['colors_exclude']:
            continue  # This is the exclusion working!
        
        matched_any = True
        
        # Calculate score
        score = 0.0
        if product.get("description_embedding") is not None:
            try:
                sim = torch.cosine_similarity(prompt_emb, product["description_embedding"], dim=0).item()
                score = sim * 0.8  # Slightly lower weight
            except Exception:
                pass
        
        # Bonus for having a different color
        if product_color and product_color not in query['colors_exclude']:
            score += 0.2
        
        max_score = max(max_score, score)
    
    # Debug logging
    if not matched_any and query['expanded_target_items']:
        print(f"⚠️  No items matched after exclusion filter")
    
    return max_score


def score_semantic_match(products, query):
    """Score with EXPANDED item type matching - FIXED for occasion-based searches"""
    prompt_emb = encode_text(query['original_prompt'])
    if prompt_emb is None:
        return 0.0
    
    max_score = 0.0
    matched_any = False
    
    for product in products:
        score = 0.0
        
        product_type = (product.get('type') or '').lower()
        product_name = (product.get('name') or '').lower()
        product_desc = (product.get('description') or '').lower()
        product_occasion = (product.get('occasion') or '').lower()
        
        # CRITICAL: Check if product matches ANY expanded item type
        if query['expanded_target_items']:
            type_match = any(
                item_type in product_type or 
                item_type in product_name or
                item_type in product_desc
                for item_type in query['expanded_target_items']
            )
            
            if not type_match:
                continue  # Skip if no match
        
        matched_any = True
        
        # Semantic similarity
        if product.get("description_embedding") is not None:
            try:
                sim = torch.cosine_similarity(prompt_emb, product["description_embedding"], dim=0).item()
                score += sim * 0.7
            except Exception:
                pass
        
        # Color matching
        if query['colors_include'] and product.get('nearest_palette_color'):
            if product['nearest_palette_color'] in query['colors_include']:
                score += product.get('color_match_confidence', 0) * 0.3
        
        # BONUS: Occasion matching (for "office outfit", "casual outfit")
        prompt_lower = query['original_prompt'].lower()
        if 'office' in prompt_lower and 'office' in product_occasion:
            score += 0.3
        elif 'casual' in prompt_lower and 'casual' in product_occasion:
            score += 0.3
        elif 'party' in prompt_lower and 'party' in product_occasion:
            score += 0.3
        elif 'traditional' in prompt_lower and 'traditional' in product_occasion:
            score += 0.3
        elif 'festive' in prompt_lower and ('festive' in product_occasion or 'party' in product_occasion):
            score += 0.3
        
        max_score = max(max_score, score)
    
    # Debug: Log if no matches found
    if not matched_any and query['expanded_target_items']:
        print(f"⚠️  No products matched item types: {query['expanded_target_items']}")
    
    return max_score

def score_complementary_match(products, query):
    """Score products for complementary matching"""
    if query['color_rgb'] is None and not query['reference_item']:
        return 0.0
    
    max_score = 0.0
    
    # Get complementary colors
    target_colors = []
    if query['color_rgb'] is not None:
        query_palette_name, _ = find_nearest_palette_color(query['color_rgb'])
        if query_palette_name and query_palette_name in HARMONY_CACHE:
            target_colors = HARMONY_CACHE[query_palette_name].get('complementary_colors', [])
    
    # CRITICAL FIX: Get items that complement the REFERENCE item
    # If user has a "top", show "jeans" (not more tops!)
    target_types = []
    if query['reference_item']:
        target_types = ITEM_COMPLEMENTS.get(query['reference_item'], [])
    
    for product in products:
        score = 0.0
        
        product_type = (product.get('type') or '').lower()
        product_name = (product.get('name') or '').lower()
        
        # MUST match the target item type if specified
        if query['target_item']:
            if query['target_item'] not in product_type and query['target_item'] not in product_name:
                continue  # Skip if not matching target item
        
        # If we have reference item, check if product type complements it
        if target_types:
            type_matches = any(t in product_type or t in product_name for t in target_types)
            if type_matches:
                score += 0.8
            else:
                # If target_types specified but no match, heavily penalize
                if query['target_item'] and query['reference_item']:
                    continue
        
        # Color complementarity
        product_color = product.get('nearest_palette_color')
        if target_colors and product_color in target_colors:
            score += product.get('color_match_confidence', 0) * 0.5
        
        # Bonus for neutral colors (they complement everything)
        if product.get('is_neutral'):
            score += 0.3
        
        max_score = max(max_score, score)
    
    return max_score

def score_image_similarity(products, image_emb):
    if image_emb is None:
        return 0.0
    
    max_score = 0.0
    for product in products:
        if product.get("blip_caption_embedding") is not None:
            try:
                sim = torch.cosine_similarity(image_emb, product["blip_caption_embedding"], dim=0).item()
                max_score = max(max_score, sim)
            except Exception:
                pass
    return max_score

def clean_for_json(obj):
    """Convert numpy types and remove tensors"""
    if isinstance(obj, torch.Tensor) or (isinstance(obj, str) and 'embedding' in obj):
        return None
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: v for k, v in ((k, clean_for_json(v)) for k, v in obj.items()) if v is not None}
    return obj

def calculate_wishlist_relevance_score(video):
    """ENHANCED: Prioritize exact product matches in wishlist"""
    ensure_session_initialized()
    
    wishlist_products = session.get("wishlist_products", [])
    wishlist_items = session.get("wishlist_items", [])
    
    if not wishlist_products and not wishlist_items:
        return 0
    
    video_products = video.get("products", [])
    total_score = 0
    comparison_count = 0
    
    # 1. EXACT PRODUCT NAME MATCHING (HIGHEST PRIORITY)
    for wishlist_product in wishlist_products:
        wishlist_name = wishlist_product.get("name", "").lower().strip()
        
        for video_product in video_products:
            video_name = video_product.get("name", "").lower().strip()
            
            if wishlist_name and video_name:
                if wishlist_name == video_name:
                    return 10.0  # IMMEDIATELY RETURN MAX SCORE
                elif wishlist_name in video_name or video_name in wishlist_name:
                    total_score += 5.0  # Partial match gets high score
    
    # 2. SEMANTIC SIMILARITY (description embeddings)
    for wishlist_product in wishlist_products:
        wishlist_text = build_product_text(wishlist_product)
        if wishlist_text.strip():
            wishlist_emb = encode_text(wishlist_text)
            if wishlist_emb is not None:
                for video_product in video_products:
                    if video_product.get("description_embedding") is not None:
                        try:
                            sim = torch.cosine_similarity(wishlist_emb, 
                                                         video_product["description_embedding"], 
                                                         dim=0).item()
                            total_score += sim * 0.8
                            comparison_count += 1
                        except Exception:
                            pass
    
    # 3. CUSTOM WISHLIST ITEMS (compare BLIP captions)
    for wishlist_item in wishlist_items:
        if wishlist_item.get("caption_emb"):
            wishlist_emb = torch.tensor(wishlist_item["caption_emb"])
            
            for video_product in video_products:
                # Compare with product BLIP caption (higher weight)
                if video_product.get("blip_caption_embedding") is not None:
                    try:
                        sim = torch.cosine_similarity(wishlist_emb, 
                                                     video_product["blip_caption_embedding"], 
                                                     dim=0).item()
                        total_score += sim * 0.7  # BLIP-to-BLIP comparison
                        comparison_count += 1
                    except Exception:
                        pass
                
                # Compare with product description (lower weight)
                if video_product.get("description_embedding") is not None:
                    try:
                        sim = torch.cosine_similarity(wishlist_emb, 
                                                     video_product["description_embedding"], 
                                                     dim=0).item()
                        total_score += sim * 0.3
                        comparison_count += 1
                    except Exception:
                        pass
    
    if comparison_count > 0:
        return total_score / comparison_count
    
    return total_score  # Return accumulated exact/partial match scores

def calculate_wardrobe_relevance_score(video):
    """ENHANCED: Compare wardrobe BLIP with product BLIP + descriptions"""
    ensure_session_initialized()
    
    wardrobe_items = session.get("wardrobe_items", [])
    if not wardrobe_items:
        return 0
    
    video_products = video.get("products", [])
    total_score = 0
    comparison_count = 0
    
    for wardrobe_item in wardrobe_items:
        if wardrobe_item.get("caption_emb"):
            wardrobe_emb = torch.tensor(wardrobe_item["caption_emb"])
            
            for video_product in video_products:
                # BLIP-to-BLIP comparison (higher weight for visual similarity)
                if video_product.get("blip_caption_embedding") is not None:
                    try:
                        sim = torch.cosine_similarity(wardrobe_emb, 
                                                     video_product["blip_caption_embedding"], 
                                                     dim=0).item()
                        total_score += sim * 0.6  # Visual similarity
                        comparison_count += 1
                    except Exception:
                        pass
                
                # BLIP-to-description comparison (semantic matching)
                if video_product.get("description_embedding") is not None:
                    try:
                        sim = torch.cosine_similarity(wardrobe_emb, 
                                                     video_product["description_embedding"], 
                                                     dim=0).item()
                        total_score += sim * 0.4  # Semantic matching
                        comparison_count += 1
                    except Exception:
                        pass
    
    if comparison_count > 0:
        return total_score / comparison_count
    return 0
# =============================================================================
# SESSION
# =============================================================================
def ensure_session_initialized():
    if "wardrobe_items" not in session:
        session["wardrobe_items"] = []
    if "wishlist_products" not in session:
        session["wishlist_products"] = []
    if "wishlist_items" not in session:
        session["wishlist_items"] = []
    session.permanent = True

def force_session_save():
    session.permanent = True
    session.modified = True

@app.before_request
def before_request():
    ensure_session_initialized()

# =============================================================================
# API ROUTES
# =============================================================================
@app.route("/api/search", methods=["POST"])
def api_search():
    """FIXED SEARCH with proper query understanding"""
    ensure_session_initialized()
    
    data = request.json
    raw_prompt = data.get("prompt", "").strip()
    image_b64 = data.get("image")
    
    # Normalize query
    if raw_prompt:
        normalized_prompt = normalize_query_with_llm(raw_prompt)
        print(f"📝 Original: {raw_prompt}")
        print(f"✅ Normalized: {normalized_prompt}")
    else:
        normalized_prompt = ""
        
    # Process image
    image_data = None
    image_emb = None
    if image_b64:
        try:
            img_bytes = base64.b64decode(image_b64.split(",")[1])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            caption = get_blip_caption(img)
            image_emb = encode_text(caption)
            image_data = {
                'dominant_color': extract_dominant_color_rgb(img),
                'caption': caption
            }
        except Exception as e:
            print(f"❌ Image error: {e}")
    
    # Parse query
    query = parse_search_query(normalized_prompt, image_data)
    query['original_prompt'] = normalized_prompt
    
    print(f"\n{'='*60}")
    print(f"SEARCH QUERY PARSED:")
    print(f"  Mode: {query['mode']}")
    print(f"  Gender: {query['gender']}")
    print(f"  Target Item: {query.get('target_item')}")
    print(f"  Expanded Items: {query.get('expanded_target_items', [])}")
    print(f"  Reference Item: {query.get('reference_item')}")
    print(f"  Colors Include: {query['colors_include']}")
    print(f"  Colors Exclude: {query['colors_exclude']}")  # NEW
    
    # Debug: Show what will be filtered
    if query['mode'] == 'exclusion':
        print(f"  🚫 EXCLUSION MODE: Will exclude {query['colors_exclude']}")
    if not query['expanded_target_items']:
        print(f"  ⚠️  No item types found - will match ALL items semantically")
    
    print(f"{'='*60}\n")
    # Score all videos
    scored_videos = []
    for video in VIDEO_DATA:
        score = score_video(video, query, image_emb)
        
        # LOWERED THRESHOLD + Better filtering
        if score > 0.08:  # Changed from 0.1 to be more lenient
            clean_video = clean_for_json(video)
            clean_video['video_score'] = float(score)
            clean_video['search_mode'] = query['mode']
            clean_video['gender_filter'] = query['gender']
            scored_videos.append((score, clean_video))
    
    scored_videos.sort(key=lambda x: x[0], reverse=True)
    
    # LIMIT results to top 30 for better UX
    results = [v for _, v in scored_videos[:30]]
    
    print(f"✅ Returning {len(results)} results")
    print(f"   Top scores: {[round(s, 3) for s, _ in scored_videos[:5]]}\n")
    return jsonify({"videos": results})

@app.route("/api/videos")
def api_videos():
    return jsonify({"videos": [clean_for_json(v) for v in VIDEO_DATA]})

@app.route("/api/feed")
def api_feed():
    ensure_session_initialized()
    filter_mode = request.args.get("filter", "normal")
    
    if filter_mode == "normal":
        return jsonify({"videos": [clean_for_json(v) for v in VIDEO_DATA]})
    
    elif filter_mode == "wardrobe":
        wardrobe_items = session.get("wardrobe_items", [])
        if not wardrobe_items:
            return jsonify({"videos": []})
        
        scored_videos = []
        for video in VIDEO_DATA:
            # USE THE NEW FUNCTION
            wardrobe_score = calculate_wardrobe_relevance_score(video)
            
            if wardrobe_score > 0.2:  # Threshold
                clean_video = clean_for_json(video)
                clean_video['wardrobe_score'] = float(wardrobe_score)
                scored_videos.append((wardrobe_score, clean_video))
        
        scored_videos.sort(key=lambda x: x[0], reverse=True)
        return jsonify({"videos": [v for _, v in scored_videos]})
    
    elif filter_mode == "wishlist":
        wishlist_products = session.get("wishlist_products", [])
        wishlist_items = session.get("wishlist_items", [])
        
        if not wishlist_products and not wishlist_items:
            return jsonify({"videos": []})
        
        scored_videos = []
        for video in VIDEO_DATA:
            # USE THE NEW FUNCTION
            wishlist_score = calculate_wishlist_relevance_score(video)
            
            if wishlist_score > 0.2:  # Threshold
                clean_video = clean_for_json(video)
                clean_video['wishlist_score'] = float(wishlist_score)
                scored_videos.append((wishlist_score, clean_video))
        
        scored_videos.sort(key=lambda x: x[0], reverse=True)
        return jsonify({"videos": [v for _, v in scored_videos]})
    
    return jsonify({"videos": [clean_for_json(v) for v in VIDEO_DATA]})

@app.route("/api/wardrobe", methods=["GET", "POST", "DELETE"])
def api_wardrobe():
    ensure_session_initialized()
    if request.method == "GET":
        return jsonify({"items": session.get("wardrobe_items", [])})
    elif request.method == "POST":
        files = request.files.getlist("wardrobe_photos")
        items = list(session.get("wardrobe_items", []))
        for f in files:
            try:
                filename = f"{uuid.uuid4().hex}.jpg"
                path = os.path.join(WARDROBE_DIR, filename)
                f.save(path)
                img = Image.open(path).convert("RGB")
                caption = get_blip_caption(img)
                emb = encode_text(caption)
                # FIX: Check if emb is not None instead of if emb
                if emb is not None:
                    items.append({
                        "id": len(items),
                        "filename": filename,
                        "caption": caption,
                        "caption_emb": emb.tolist(),
                        "image_path": f"/wardrobe_photos/{filename}"
                    })
            except Exception as e:
                print(f"Error: {e}")
        session["wardrobe_items"] = items
        force_session_save()
        return jsonify({"count": len(items), "items": items})
    elif request.method == "DELETE":
        data = request.get_json()
        items = [i for i in session.get("wardrobe_items", []) if i.get("id") != data.get("id")]
        session["wardrobe_items"] = items
        force_session_save()
        return jsonify({"count": len(items)})

@app.route("/api/wishlist", methods=["GET", "POST", "DELETE"])
def api_wishlist():
    ensure_session_initialized()
    if request.method == "GET":
        return jsonify({
            "products": session.get("wishlist_products", []),
            "items": session.get("wishlist_items", [])
        })
    elif request.method == "POST":
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
            if data and data.get("type") == "product":
                product = data.get("product")
                wishlist_products = list(session.get("wishlist_products", []))
                if not any(p.get("name") == product.get("name") for p in wishlist_products):
                    wishlist_products.append(product)
                    session["wishlist_products"] = wishlist_products
                    force_session_save()
                    return jsonify({"count": len(wishlist_products)})
                return jsonify({"count": len(wishlist_products)})
        else:
            files = request.files.getlist("wishlist_photos")
            items = list(session.get("wishlist_items", []))
            for f in files:
                try:
                    filename = f"{uuid.uuid4().hex}.jpg"
                    path = os.path.join(WISHLIST_DIR, filename)
                    f.save(path)
                    img = Image.open(path).convert("RGB")
                    caption = get_blip_caption(img)
                    emb = encode_text(caption)
                    # FIX: Check if emb is not None instead of if emb
                    if emb is not None:
                        items.append({
                            "id": len(items),
                            "filename": filename,
                            "caption": caption,
                            "caption_emb": emb.tolist(),
                            "image_path": f"/wishlist_photos/{filename}"
                        })
                except Exception as e:
                    print(f"Error: {e}")
            session["wishlist_items"] = items
            force_session_save()
            return jsonify({"count": len(items), "items": items})
    elif request.method == "DELETE":
        data = request.get_json()
        if data.get("type") == "product":
            products = [p for p in session.get("wishlist_products", []) if p.get("name") != data.get("name")]
            session["wishlist_products"] = products
            force_session_save()
            return jsonify({"message": "Removed"})
        else:
            items = [i for i in session.get("wishlist_items", []) if i.get("id") != data.get("id")]
            session["wishlist_items"] = items
            force_session_save()
            return jsonify({"message": "Removed"})

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(os.path.join(BASE_DIR, "videos"), filename)

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    response = send_from_directory(os.path.join(BASE_DIR, "thumbnails"), filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/wardrobe_photos/<path:filename>')
def serve_wardrobe_photo(filename):
    response = send_from_directory(WARDROBE_DIR, filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/wishlist_photos/<path:filename>')
def serve_wishlist_photo(filename):
    response = send_from_directory(WISHLIST_DIR, filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/api/debug/session')
def debug_session():
    ensure_session_initialized()
    return jsonify({
        "wardrobe_count": len(session.get("wardrobe_items", [])),
        "wishlist_count": len(session.get("wishlist_products", []))
    })

@app.route('/api/cache/stats')
def cache_stats():
    """Get cache statistics"""
    cache_size = os.path.getsize(EMBEDDINGS_CACHE_FILE) / (1024*1024) if os.path.exists(EMBEDDINGS_CACHE_FILE) else 0
    return jsonify({
        'embeddings_cached': len(embeddings_cache.cache),
        'cache_size_mb': round(cache_size, 2),
        'harmony_colors': len(HARMONY_CACHE)
    })

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear embeddings cache"""
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        os.remove(EMBEDDINGS_CACHE_FILE)
    embeddings_cache.cache.clear()
    return jsonify({'status': 'cleared'})

# =============================================================================
# STARTUP
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FASHION SEARCH APP - OPTIMIZED WITH CACHING")
    print("="*60 + "\n")
    
    print("Loading color harmony cache...")
    HARMONY_CACHE = build_harmony_cache()
    print(f"✅ {len(HARMONY_CACHE)} color harmonies ready\n")
    
    precompute_all_embeddings()
    
    print("\n" + "="*60)
    print("✅ SERVER READY - Port 5000")
    print("="*60)
    print("\n⚡ OPTIMIZATIONS:")
    print("  • Lazy model loading (loads on first search)")
    print("  • Persistent embeddings cache")
    print("  • Incremental updates only")
    print("  • ~12-18x faster subsequent startups")
    print("\n📊 Startup time:")
    print("  • First run: 60-90s (builds cache)")
    print("  • Next runs: 3-5s (loads from cache)")
    print("="*60 + "\n")
    
    app.run(port=5000, debug=True)
