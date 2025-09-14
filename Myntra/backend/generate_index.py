import os, json, subprocess
from pathlib import Path

BASE = Path(__file__).parent
VIDEO_DIR = BASE / "videos"
THUMB_DIR = BASE / "thumbnails"
OUT = BASE / "video_index.json"

THUMB_DIR.mkdir(exist_ok=True)

videos = []
for mp4 in VIDEO_DIR.glob("*.mp4"):
    thumb = THUMB_DIR / f"{mp4.stem}.jpg"
    if not thumb.exists():
        # Generate thumbnail at 5 seconds
        subprocess.run([
            "ffmpeg", "-ss", "00:00:05", "-i", str(mp4),
            "-frames:v", "1", "-q:v", "2", str(thumb)
        ])
    videos.append({
        "id": mp4.stem,
        "videoUrl": f"/media/videos/{mp4.name}",
        "tags": ["monsoon", "casual"] if "8970055" in mp4.name else ["summer"],  # demo tags
        "thumbnail": f"/media/thumbnails/{mp4.stem}.jpg",
        "suggestions": [
            {"id": "p1", "title": "Demo Jacket", "price": "₹1999", "imageUrl": f"/media/thumbnails/{mp4.stem}.jpg"}
        ]
    })

with open(OUT, "w") as f:
    json.dump({"videos": videos}, f, indent=2)

print("Generated", OUT)
