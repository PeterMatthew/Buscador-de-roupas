"use client";
import React, { useState, useRef, useEffect } from "react";
import { useDetection } from "./hooks/useDetection";
import { useSearch } from "./hooks/useSearch";
import { ImageUpload } from "./components/ImageUpload";
import { ProductGrid } from "./components/ProductGrid";
import DetectedItem from "./components/DetectedItem";
import { Detection } from "./types";

const SUPPORTED_CATEGORIES = ["short_sleeve_top"];

const Home = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(
    null,
  );

  const { detections, isLoading, detectItems, setDetections } = useDetection();
  const {
    similarItems,
    pagination,
    isSearching,
    temperature,
    setTemperature,
    search,
    setSimilarItems,
    setPagination,
  } = useSearch();

  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (imageFile) detectItems(imageFile);
  }, [imageFile]);

  const handleUpload = (file: File) => {
    setImageFile(file);
    setImageSrc(URL.createObjectURL(file));
    setDetections(null);
    setSimilarItems([]);
    setSelectedDetection(null);
    setPagination(null);
  };

  const handleSearch = (
    detection: Detection,
    offset: number = 0,
    tempOverride?: number,
  ) => {
    if (!imageFile) return;
    // Guardrail: Check if category is supported
    if (!SUPPORTED_CATEGORIES.includes(detection.category)) {
      setSelectedDetection(detection);
      setSimilarItems([]);
      setPagination(null);
      return;
    }

    setSelectedDetection(detection);
    search(imageFile, detection, offset, tempOverride);
  };

  const handleTemperatureChange = (newTemp: number) => {
    setTemperature(newTemp);
    // Re-trigger search with the selected detection if one is active
    if (
      selectedDetection &&
      SUPPORTED_CATEGORIES.includes(selectedDetection.category)
    ) {
      handleSearch(selectedDetection, 0, newTemp);
    }
  };

  return (
    <div className="min-h-screen bg-white py-8">
      <div className="max-w-6xl mx-auto px-4">
        <header className="text-center mb-6">
          <h1 className="text-4xl font-serif text-gray-900">Clothing Finder</h1>
        </header>

        <ImageUpload onUpload={handleUpload} />

        <main className="flex flex-col items-center mt-6">
          {imageSrc && (
            <div className="relative bg-gray-50 max-w-96 border">
              <img
                ref={imageRef}
                src={imageSrc}
                alt="Input"
                className="w-full h-auto"
              />

              {/* Overlay Detections */}
              {detections?.map((det, idx) => (
                <DetectedItem
                  key={idx}
                  bbox={det.box}
                  imageWidth={imageRef.current?.naturalWidth || 0}
                  imageHeight={imageRef.current?.naturalHeight || 0}
                  category={det.category}
                  isSelected={selectedDetection?.box === det.box}
                  onClick={() => handleSearch(det, 0)}
                />
              ))}

              {isLoading && (
                <div className="absolute inset-0 bg-white/80 flex flex-col items-center justify-center">
                  <div className="w-6 h-6 border-2 border-black border-t-transparent rounded-full animate-spin mb-2" />
                  <p className="text-sm">Analyzing...</p>
                </div>
              )}
            </div>
          )}

          {/* Temperature Control Slider */}
          {selectedDetection &&
            SUPPORTED_CATEGORIES.includes(selectedDetection.category) && (
              <div className="w-full max-w-md my-6 p-4 border rounded-lg bg-gray-50 flex flex-col items-center gap-2">
                <div className="flex justify-between w-full text-xs font-medium text-gray-700">
                  <span>Exact match</span>
                  <span className="font-semibold text-black">
                    Temperature: {temperature.toFixed(2)}
                  </span>
                  <span>Exploration</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="1.0"
                  step="0.05"
                  value={temperature}
                  onChange={(e) =>
                    handleTemperatureChange(parseFloat(e.target.value))
                  }
                  className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-black"
                />
              </div>
            )}

          {/* Result Section */}
          <section className="w-full mt-4">
            {isSearching ? (
              <div className="text-center py-16">
                <div className="w-8 h-8 border-2 border-black border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <p>Searching for similar items...</p>
              </div>
            ) : similarItems.length > 0 ? (
              <ProductGrid
                items={similarItems}
                pagination={pagination}
                onNextPage={() =>
                  selectedDetection &&
                  handleSearch(selectedDetection, pagination?.next_offset!)
                }
                onPrevPage={() =>
                  selectedDetection &&
                  handleSearch(selectedDetection, pagination?.prev_offset!)
                }
              />
            ) : (
              <div className="text-center py-16 border border-dashed text-gray-600">
                {!imageSrc
                  ? "Upload an image to begin"
                  : "Select a detected item to search"}
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
};

export default Home;
