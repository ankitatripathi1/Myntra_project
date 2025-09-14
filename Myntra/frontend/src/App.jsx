import React, { useState, useEffect, useRef } from "react";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [imagePreview, setImagePreview] = useState(null);
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showProduct, setShowProduct] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [likeCounts, setLikeCounts] = useState({});
  const [liked, setLiked] = useState({});
  const [showComment, setShowComment] = useState(false);
  const [commentVideoId, setCommentVideoId] = useState(null);
  const [copied, setCopied] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const videoRefs = useRef([]);
  const fileInputRef = useRef(null);

  // Load videos
  useEffect(() => {
    fetch("http://localhost:5000/api/videos")
      .then((res) => res.json())
      .then((data) => setVideos(data.videos));
  }, []);

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);
  }

  // Handle Search
  async function handleSearch(e) {
    e.preventDefault();
    setLoading(true);
    const payload = { prompt, image: imagePreview, max_price: maxPrice || null };

    const res = await fetch("http://localhost:5000/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    setVideos(data.videos);
    setCurrentIndex(0);
    setLoading(false);

    setTimeout(() => {
      if (videoRefs.current[0]) {
        videoRefs.current[0].scrollIntoView({ behavior: "smooth" });
      }
    }, 100);
  }

  // Product Modal
  function ProductModal({ product, onClose }) {
    if (!product) return null;
    const images = product.image_url ? [product.image_url] : [];
    return (
      <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl p-6 w-full max-w-xs relative">
          <button onClick={onClose} className="absolute top-2 right-2 text-xl">✖️</button>
          {images.length > 0 && (
            <div className="mb-3 flex flex-col items-center">
              <img src={images[0]} alt={product.name} className="w-full h-40 object-cover rounded mb-2" />
            </div>
          )}
          <div className="font-bold text-lg mb-1">{product.name}</div>
          <div className="text-gray-700 mb-2">₹{product.price}</div>
          <div className="text-sm text-gray-500 mb-2">{product.occasion || "No occasion specified."}</div>
          <a href={product.url || "#"} target="_blank" rel="noopener noreferrer" className="block mt-2 px-4 py-2 bg-indigo-600 text-white rounded text-center">
            Shop Now
          </a>
        </div>
      </div>
    );
  }

  // Comment Modal
  function CommentModal({ onClose }) {
    return (
      <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl p-6 w-full max-w-sm relative">
          <button onClick={onClose} className="absolute top-2 right-2 text-xl">✖️</button>
          <div className="font-bold mb-2">Comments</div>
          <div className="text-gray-500 mb-4">(Comment functionality coming soon)</div>
        </div>
      </div>
    );
  }

  // Like/Comment/Share
  function handleLike(videoId) {
    setLiked((prev) => {
      const isLiked = !prev[videoId];
      setLikeCounts((counts) => ({
        ...counts,
        [videoId]: (counts[videoId] || 0) + (isLiked ? 1 : -1),
      }));
      return { ...prev, [videoId]: isLiked };
    });
  }
  function handleComment(videoId) {
    setCommentVideoId(videoId);
    setShowComment(true);
  }
  function handleShare(videoId) {
    const url = window.location.href + `?video=${videoId}`;
    navigator.clipboard.writeText(url);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  }
  function handleProductClick(product) {
    setSelectedProduct(product);
    setShowProduct(true);
  }

  // Scroll handling
  function handleScroll(e) {
    const scrollTop = e.target.scrollTop;
    const vh = window.innerHeight;
    const idx = Math.round(scrollTop / vh);
    if (idx !== currentIndex) setCurrentIndex(idx);
  }
  useEffect(() => {
    if (videoRefs.current[currentIndex]) {
      videoRefs.current[currentIndex].scrollIntoView({ behavior: "smooth" });
    }
  }, [currentIndex]);

  // 🔥 Skeleton Loader Component
  function SkeletonCard() {
    return (
      <div className="h-screen flex items-center justify-center bg-black">
        <div className="animate-pulse w-[min(100vw,430px)] h-[76vh] sm:h-[90vh] bg-gray-800 rounded-xl flex flex-col">
          <div className="flex-1 bg-gray-700 rounded-xl" />
          <div className="flex gap-3 mt-3 p-2 overflow-x-auto">
            {[1, 2, 3].map((i) => (
              <div key={i} className="w-[100px] h-[90px] bg-gray-600 rounded" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full bg-black flex flex-col">
      {/* SEARCH BAR */}
      <div className="w-full max-w-3xl mx-auto z-20">
        <form onSubmit={handleSearch} className="bg-white rounded-2xl shadow p-4 mb-4 mt-2">
          <div className="flex gap-3 items-center">
            <div className="flex-1 flex flex-col gap-2">
              <input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Search outfits (e.g., 'gold dress party')" className="w-full px-4 py-3 rounded-lg border focus:outline-none" />
              <input type="number" value={maxPrice} onChange={(e) => setMaxPrice(e.target.value)} placeholder="Max Price" className="w-full px-4 py-2 rounded-lg border focus:outline-none" />
            </div>
            <div className="flex items-center gap-2">
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              <button type="button" onClick={() => fileInputRef.current.click()} className="px-4 py-2 bg-indigo-600 text-white rounded-lg">Upload</button>
              <button type="submit" className="px-4 py-2 bg-green-600 text-white rounded-lg">Search</button>
            </div>
          </div>
        </form>
      </div>

      {/* FEED */}
      <div className="flex-1 overflow-y-scroll snap-y snap-mandatory" style={{ WebkitOverflowScrolling: "touch" }} onScroll={handleScroll}>
        {loading
          ? Array(3).fill(0).map((_, i) => <SkeletonCard key={i} />) // 👈 Show 3 shimmer cards
          : videos.map((video, idx) => (
              <div key={video.video_id} ref={(el) => (videoRefs.current[idx] = el)} className="relative h-screen w-full flex items-center justify-center snap-start bg-black" style={{ minHeight: "100vh" }}>
                <div className="relative mx-auto w-[min(100vw,430px)] h-[76vh] sm:h-[90vh] flex items-center justify-center bg-black rounded-xl overflow-hidden shadow-2xl">
                  <video className="w-full h-full object-cover bg-black" src={`http://localhost:5000${video.video_url}`} controls loop playsInline autoPlay={idx === currentIndex} muted style={{ aspectRatio: "9/16", background: "black" }} />
                </div>

                {/* PRODUCT STRIP */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-[90vw] max-w-2xl bg-black/30 py-2 px-3 rounded-xl backdrop-blur-md flex gap-2 overflow-x-auto z-10 border border-white/10" style={{ pointerEvents: "auto" }}>
                  {video.products.map((p, i) => (
                    <div key={p.name + "-" + i} className="min-w-[110px] bg-white/30 rounded p-2 flex-shrink-0 cursor-pointer hover:scale-105 transition-transform border border-white/20" onClick={() => handleProductClick(p)}>
                      {p.image_url && <img src={p.image_url} alt={p.name} className="w-full h-16 object-cover rounded" />}
                      <div className="text-xs mt-1 font-semibold text-white drop-shadow">{p.name}</div>
                      <div className="text-xs text-gray-200">₹{p.price}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
      </div>

      {showProduct && <ProductModal product={selectedProduct} onClose={() => setShowProduct(false)} />}
      {showComment && <CommentModal onClose={() => setShowComment(false)} />}
    </div>
  );
}
