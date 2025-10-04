import React, { useState, useEffect, useRef } from "react";
import { User, Heart, ShoppingBag, Menu, RefreshCw, Plus, X, Upload, Search as SearchIcon } from "lucide-react";

const apiCall = async (url, options = {}) => {
  const config = {
    credentials: 'include',
    ...options,
    headers: {
      ...options.headers,
    }
  };
  
  try {
    const response = await fetch(url, config);
    return response;
  } catch (error) {
    console.error(`API call failed for ${url}:`, error);
    throw error;
  }
};

const WardrobePage = ({ onBack }) => {
  const [wardrobeItems, setWardrobeItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadWardrobeItems();
  }, []);

  async function loadWardrobeItems() {
    try {
      setLoading(true);
      const res = await apiCall("http://localhost:5000/api/wardrobe");
      const data = await res.json();
      setWardrobeItems(data.items || []);
    } catch (error) {
      console.error("Failed to load wardrobe:", error);
    } finally {
      setLoading(false);
    }
  }

  async function uploadFiles(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    setUploading(true);
    
    try {
      const formData = new FormData();
      files.forEach(f => formData.append("wardrobe_photos", f));

      const res = await apiCall("http://localhost:5000/api/wardrobe", {
        method: "POST",
        body: formData,
      });
      
      if (!res.ok) throw new Error(`Upload failed`);
      
      const data = await res.json();
      if (data.items) {
        setWardrobeItems(data.items);
      } else {
        await loadWardrobeItems();
      }
      
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function deleteItem(id) {
    if (!confirm("Remove this item?")) return;
    
    try {
      await apiCall("http://localhost:5000/api/wardrobe", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
      });
      
      setWardrobeItems(prev => prev.filter(item => item.id !== id));
    } catch (error) {
      console.error("Delete failed:", error);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header with Glassmorphism */}
      <div className="sticky top-0 z-20 backdrop-blur-lg bg-white/80 shadow-lg px-6 py-4 flex items-center justify-between border-b border-gray-200/50">
        <button
          onClick={onBack}
          className="px-5 py-2.5 bg-gradient-to-r from-pink-500 to-pink-600 text-white rounded-xl hover:from-pink-600 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-pink-500/50 font-medium"
        >
          ← Back
        </button>
        <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-600 to-purple-600 bg-clip-text text-transparent">
          My Wardrobe ({wardrobeItems.length})
        </h1>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-5 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-blue-500/50 flex items-center gap-2 font-medium"
          disabled={uploading}
        >
          <Plus size={18} />
          {uploading ? "Uploading..." : "Add Items"}
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={uploadFiles}
        className="hidden"
      />

      <div className="p-8">
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-pink-500 border-t-transparent"></div>
          </div>
        ) : wardrobeItems.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-pink-100 to-purple-100 rounded-full flex items-center justify-center">
              <ShoppingBag size={40} className="text-pink-600" />
            </div>
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Your wardrobe is empty</h3>
            <p className="text-gray-500 mb-6">Start building your digital closet</p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-8 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-300 shadow-xl hover:shadow-blue-500/50 flex items-center gap-2 mx-auto font-medium"
            >
              <Upload size={20} />
              Upload Your First Items
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {wardrobeItems.map((item) => (
              <div key={`wardrobe-${item.id}`} className="group relative bg-white rounded-2xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden transform hover:-translate-y-1">
                <div className="aspect-square overflow-hidden bg-gray-100">
                  <img
                    src={`http://localhost:5000${item.image_path}`}
                    alt={item.caption}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                </div>
                <div className="p-4">
                  <p className="text-sm text-gray-700 line-clamp-2">{item.caption}</p>
                </div>
                <button
                  onClick={() => deleteItem(item.id)}
                  className="absolute top-3 right-3 bg-red-500 text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition-all duration-300 shadow-lg hover:bg-red-600 hover:scale-110"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const WishlistPage = ({ onBack, refreshSignal }) => {
  const [wishlistProducts, setWishlistProducts] = useState([]);
  const [wishlistItems, setWishlistItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadWishlistData();
  }, [refreshSignal]);

  async function loadWishlistData() {
    try {
      setLoading(true);
      const res = await apiCall("http://localhost:5000/api/wishlist");
      const data = await res.json();
      setWishlistProducts(data.products || []);
      setWishlistItems(data.items || []);
    } catch (error) {
      console.error("Failed to load wishlist:", error);
    } finally {
      setLoading(false);
    }
  }

  async function uploadFiles(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    setUploading(true);
    
    try {
      const formData = new FormData();
      files.forEach(f => formData.append("wishlist_photos", f));

      const res = await apiCall("http://localhost:5000/api/wishlist", {
        method: "POST",
        body: formData,
      });
      
      if (!res.ok) throw new Error(`Upload failed`);
      
      const data = await res.json();
      if (data.items) {
        setWishlistItems(data.items);
      } else {
        await loadWishlistData();
      }
      
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function deleteProduct(name) {
    if (!confirm(`Remove "${name}"?`)) return;
    
    try {
      await apiCall("http://localhost:5000/api/wishlist", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "product", name }),
      });
      
      setWishlistProducts(prev => prev.filter(p => p.name !== name));
    } catch (error) {
      console.error("Delete failed:", error);
    }
  }

  async function deleteItem(id) {
    if (!confirm("Remove this item?")) return;
    
    try {
      await apiCall("http://localhost:5000/api/wishlist", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id }),
      });
      
      setWishlistItems(prev => prev.filter(item => item.id !== id));
    } catch (error) {
      console.error("Delete failed:", error);
    }
  }

  const totalItems = wishlistProducts.length + wishlistItems.length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="sticky top-0 z-20 backdrop-blur-lg bg-white/80 shadow-lg px-6 py-4 flex items-center justify-between border-b border-gray-200/50">
        <button
          onClick={onBack}
          className="px-5 py-2.5 bg-gradient-to-r from-pink-500 to-pink-600 text-white rounded-xl hover:from-pink-600 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-pink-500/50 font-medium"
        >
          ← Back
        </button>
        <h1 className="text-2xl font-bold bg-gradient-to-r from-pink-600 to-red-600 bg-clip-text text-transparent">
          My Wishlist ({totalItems})
        </h1>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-5 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-blue-500/50 flex items-center gap-2 font-medium"
          disabled={uploading}
        >
          <Plus size={18} />
          {uploading ? "Uploading..." : "Add Photos"}
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={uploadFiles}
        className="hidden"
      />

      <div className="p-8 space-y-10">
        {/* Wishlisted Products */}
        <div>
          <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
            <Heart className="text-pink-500" size={24} />
            Wishlisted Products ({wishlistProducts.length})
          </h2>
          {wishlistProducts.length === 0 ? (
            <div className="text-center py-12 bg-white rounded-2xl shadow-md">
              <p className="text-gray-500">Click the heart icon on video products to add them!</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {wishlistProducts.map((product, idx) => (
                <div key={`product-${idx}`} className="group relative bg-white rounded-2xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden transform hover:-translate-y-1">
                  {product.image_url && (
                    <div className="aspect-square overflow-hidden bg-gray-100">
                      <img
                        src={product.image_url.startsWith("http") ? product.image_url : `http://localhost:5000${product.image_url}`}
                        alt={product.name}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                      />
                    </div>
                  )}
                  <div className="p-4">
                    <h3 className="font-semibold text-sm line-clamp-2 mb-1">{product.name}</h3>
                    <p className="text-pink-600 font-bold text-lg">₹{product.price}</p>
                    {product.color && <p className="text-gray-500 text-xs mt-1">{product.color}</p>}
                  </div>
                  <button
                    onClick={() => deleteProduct(product.name)}
                    className="absolute top-3 right-3 bg-red-500 text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition-all duration-300 shadow-lg hover:bg-red-600 hover:scale-110"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Custom Items */}
        <div>
          <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
            <Upload className="text-blue-500" size={24} />
            Custom Items ({wishlistItems.length})
          </h2>
          {wishlistItems.length === 0 ? (
            <div className="text-center py-16 bg-white rounded-2xl shadow-md">
              <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full flex items-center justify-center">
                <Upload size={32} className="text-blue-600" />
              </div>
              <p className="text-gray-500 mb-6">No custom items yet</p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-8 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-300 shadow-xl hover:shadow-blue-500/50 inline-flex items-center gap-2 font-medium"
              >
                <Upload size={20} />
                Upload Items You Want
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {wishlistItems.map((item) => (
                <div key={`item-${item.id}`} className="group relative bg-white rounded-2xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden transform hover:-translate-y-1">
                  <div className="aspect-square overflow-hidden bg-gray-100">
                    <img
                      src={`http://localhost:5000${item.image_path}`}
                      alt={item.caption}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                    />
                  </div>
                  <div className="p-4">
                    <p className="text-sm text-gray-700 line-clamp-2">{item.caption}</p>
                  </div>
                  <button
                    onClick={() => deleteItem(item.id)}
                    className="absolute top-3 right-3 bg-red-500 text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition-all duration-300 shadow-lg hover:bg-red-600 hover:scale-110"
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const [currentPage, setCurrentPage] = useState("feed");
  const [prompt, setPrompt] = useState("");
  const [imagePreview, setImagePreview] = useState(null);
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filterMode, setFilterMode] = useState("normal");
  const [wishlistVersion, setWishlistVersion] = useState(0);

  const videoRefs = useRef([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (currentPage === "feed") {
      loadFeed();
    }
  }, [currentPage, filterMode]);

  async function loadFeed() {
    try {
      const res = await apiCall(`http://localhost:5000/api/feed?filter=${filterMode}`);
      const data = await res.json();
      setVideos(data.videos);
    } catch (error) {
      console.error("Failed to load feed:", error);
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);
  }

  async function handleSearch() {
    setLoading(true);

    const payload = {
      prompt,
      image: imagePreview,
      filter_mode: filterMode === "normal" ? null : filterMode,
    };

    try {
      const res = await apiCall("http://localhost:5000/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      
      if (!res.ok) throw new Error(`Search failed`);
      
      const data = await res.json();
      setVideos(data.videos);

      if (prompt || imagePreview) {
        setFilterMode("normal");
      }
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setLoading(false);
    }
  }

  async function addToWishlist(product) {
    try {
      const res = await apiCall("http://localhost:5000/api/wishlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "product", product }),
      });
      
      if (!res.ok) throw new Error(`Failed to add to wishlist`);
      
      setWishlistVersion(v => v + 1);
      
    } catch (error) {
      console.error("Failed to add to wishlist:", error);
    }
  }

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const video = entry.target;
          if (entry.isIntersecting) {
            video.play().catch(() => {});
          } else {
            video.pause();
          }
        });
      },
      { threshold: 0.6 }
    );

    videoRefs.current.forEach((vid) => {
      if (vid) observer.observe(vid);
    });

    return () => {
      videoRefs.current.forEach((vid) => {
        if (vid) observer.unobserve(vid);
      });
    };
  }, [videos]);

  if (currentPage === "wardrobe") {
    return <WardrobePage onBack={() => setCurrentPage("feed")} />;
  }

  if (currentPage === "wishlist") {
    return <WishlistPage onBack={() => setCurrentPage("feed")} refreshSignal={wishlistVersion} />;
  }

  return (
    <div className="h-screen w-full bg-gradient-to-br from-gray-50 to-gray-100 flex flex-col">
      {/* Enhanced Header with Glassmorphism */}
      <header className="sticky top-0 z-30 backdrop-blur-lg bg-white/90 shadow-lg px-6 py-2 flex items-center justify-between border-b border-gray-200/50">

        <div className="font-extrabold text-3xl bg-gradient-to-r from-pink-700 to-pink-600 bg-clip-text text-transparent cursor-pointer">
          Myntra
        </div>

        <nav className="hidden md:flex gap-8 font-semibold text-gray-700 text-sm">
          {["MEN", "WOMEN", "KIDS", "HOME & LIVING", "BEAUTY"].map((item) => (
            <span
              key={item}
              className="cursor-pointer hover:text-pink-600 transition-colors duration-300 relative group"
            >
              {item}
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-pink-600 group-hover:w-full transition-all duration-300"></span>
            </span>
          ))}
        </nav>

        <div className="hidden md:flex gap-6 items-center text-sm font-medium text-gray-700">
          <div className="flex items-center gap-2 cursor-pointer hover:text-pink-600 transition-all duration-300 group">
            <User size={18} className="group-hover:scale-110 transition-transform duration-300" />
            <span>Profile</span>
          </div>
          <div
            className="flex items-center gap-2 cursor-pointer hover:text-pink-600 transition-all duration-300 group"
            onClick={() => setCurrentPage("wishlist")}
          >
            <Heart size={18} className="group-hover:scale-110 transition-transform duration-300 group-hover:fill-pink-600" />
            <span>Wishlist</span>
          </div>
          <div
            className="flex items-center gap-2 cursor-pointer hover:text-pink-600 transition-all duration-300 group"
            onClick={() => setCurrentPage("wardrobe")}
          >
            <ShoppingBag size={18} className="group-hover:scale-110 transition-transform duration-300" />
            <span>Wardrobe</span>
          </div>
        </div>

        <div className="md:hidden cursor-pointer">
          <Menu size={24} />
        </div>
      </header>

      {/* Filter Pills */}
      <div className="px-6 py-2 bg-white/80 backdrop-blur-sm shadow-sm flex gap-3 flex-wrap">

        <button
          onClick={() => setFilterMode("normal")}
          className={`px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
            filterMode === "normal"
              ? "bg-gradient-to-r from-pink-500 to-pink-600 text-white shadow-lg shadow-pink-500/50"
              : "bg-white text-gray-700 hover:bg-gray-50 shadow-md"
          }`}
        >
          All Items
        </button>
        <button
          onClick={() => setFilterMode("wardrobe")}
          className={`px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
            filterMode === "wardrobe"
              ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg shadow-blue-500/50"
              : "bg-white text-gray-700 hover:bg-gray-50 shadow-md"
          }`}
        >
          Similar to Wardrobe
        </button>
        <button
          onClick={() => setFilterMode("wishlist")}
          className={`px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 ${
            filterMode === "wishlist"
              ? "bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg shadow-green-500/50"
              : "bg-white text-gray-700 hover:bg-gray-50 shadow-md"
          }`}
        >
          Similar to Wishlist
        </button>
      </div>

      {/* Enhanced Search Bar */}
      <div className="px-6 py-2 bg-white/80 backdrop-blur-sm shadow-md">

        <div className="flex items-center gap-3">
          {imagePreview && (
            <div className="relative group">
              <img
                src={imagePreview}
                alt="preview"
                className="w-12 h-12 object-cover rounded-xl border-2 border-pink-300 shadow-md"
              />
              <button
                onClick={() => setImagePreview(null)}
                className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X size={12} />
              </button>
            </div>
          )}

          <div className="flex-1 relative">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={`Search fashion items... ${filterMode !== "normal" ? "(search will clear filter)" : ""}`}
              className="w-full px-5 py-3 pl-12 bg-white border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-pink-500 focus:border-transparent focus:outline-none text-sm transition-all duration-300 shadow-md"
              disabled={filterMode !== "normal" && !prompt && !imagePreview}
            />
            <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
          </div>

          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-5 py-3 bg-gradient-to-r from-pink-500 to-pink-600 text-white rounded-xl hover:from-pink-600 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-pink-500/50 text-sm font-medium"
          >
            Image
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />

          <button
            onClick={handleSearch}
            className="px-5 py-3 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl hover:from-green-600 hover:to-green-700 transition-all duration-300 shadow-lg hover:shadow-green-500/50 flex items-center gap-2 text-sm font-medium"
            disabled={loading}
          >
            <SearchIcon size={16} />
            {loading ? "..." : "Search"}
          </button>

          <button
            onClick={loadFeed}
            className="p-3 bg-gradient-to-r from-yellow-400 to-yellow-500 text-white rounded-xl hover:from-yellow-500 hover:to-yellow-600 transition-all duration-300 shadow-lg hover:shadow-yellow-500/50"
            title="Refresh Feed"
          >
            <RefreshCw size={18} />
          </button>
        </div>
      </div>

      {/* Video Feed with Smooth Scroll */}
      <div className="flex-1 overflow-y-scroll snap-y snap-mandatory scroll-smooth">
        {loading ? (
          <div className="h-screen flex flex-col items-center justify-center">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-pink-500 border-t-transparent mb-4"></div>
            <p className="text-gray-600 font-medium">Searching for perfect matches...</p>
          </div>
        ) : (
          videos.map((video, idx) => (
            <div
              key={video.video_id}
              className="relative h-screen w-full flex items-center justify-center snap-start bg-gradient-to-b from-gray-900 to-black"
            >
              <div className="relative mx-auto w-[min(100vw,430px)] h-[76vh] sm:h-[90vh] flex items-center justify-center bg-black rounded-2xl overflow-hidden shadow-2xl">
                <video
                  ref={(el) => (videoRefs.current[idx] = el)}
                  className="w-full h-full object-cover bg-black"
                  src={`http://localhost:5000${video.video_url}`}
                  controls={false}
                  loop
                  playsInline
                  style={{ aspectRatio: "9/16", background: "black" }}
                />

                {/* Score Badge */}
                <div className="absolute top-3 left-3 px-3 py-1.5 backdrop-blur-md bg-black/50 text-white text-xs rounded-full border border-white/20 font-medium">
                  Score: {video.video_score?.toFixed(2) || "N/A"}
                  {video.wardrobe_score && ` | W: ${video.wardrobe_score.toFixed(2)}`}
                  {video.wishlist_score && ` | Wi: ${video.wishlist_score.toFixed(2)}`}
                </div>

                {filterMode !== "normal" && (
                  <div className="absolute top-3 right-3 px-3 py-1.5 backdrop-blur-md bg-blue-500/80 text-white text-xs rounded-full border border-white/20 font-medium">
                    {filterMode === "wardrobe" ? "Wardrobe" : "Wishlist"}
                  </div>
                )}
              </div>

              {/* Product Carousel with Enhanced Design */}
              <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-[90vw] max-w-2xl backdrop-blur-xl bg-gradient-to-r from-black/40 to-black/30 py-3 px-4 rounded-2xl flex gap-3 overflow-x-auto z-10 border border-white/10 shadow-2xl scrollbar-hide">
                {video.products.map((p, i) => (
                  <div
                    key={p.name + "-" + i}
                    className="min-w-[120px] bg-white/10 backdrop-blur-md rounded-xl p-2.5 flex-shrink-0 relative group hover:bg-white/20 transition-all duration-300 border border-white/10"
                  >
                    {p.image_url && (
                      <div className="overflow-hidden rounded-lg mb-2">
                        <img
                          src={p.image_url.startsWith("http")
                            ? p.image_url
                            : `http://localhost:5000/${p.image_url}`}
                          alt={p.name}
                          className="w-full h-20 object-cover rounded-lg group-hover:scale-110 transition-transform duration-300"
                        />
                      </div>
                    )}
                    <div className="text-xs mt-1.5 font-semibold text-white line-clamp-2 leading-tight">
                      {p.name}
                    </div>
                    <div className="text-xs text-pink-300 font-bold mt-1">₹{p.price}</div>

                    <button
                      onClick={() => addToWishlist(p)}
                      className="absolute top-2 right-2 bg-gradient-to-r from-pink-500 to-red-500 text-white rounded-full p-1.5 opacity-0 group-hover:opacity-100 transition-all duration-300 shadow-lg hover:scale-110"
                      title="Add to Wishlist"
                    >
                      <Heart size={12} className="group-hover:fill-white" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      <style jsx>{`
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
        .scrollbar-hide {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
        
        /* Smooth scroll for all browsers */
        * {
          scroll-behavior: smooth;
        }
        
        /* Custom scroll snap for smoother experience */
        .snap-y {
          scroll-snap-type: y mandatory;
        }
        
        .snap-start {
          scroll-snap-align: start;
          scroll-snap-stop: always;
        }
      `}</style>
    </div>
  );
}
