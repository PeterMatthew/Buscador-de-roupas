import { useState } from "react";
import { DetectionResult } from "../types";
import { API_CONFIG } from "../config";

export const useDetection = () => {
  const [detections, setDetections] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const detectItems = async (imageFile: File) => {
    setIsLoading(true);
    setDetections(null);

    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const url = `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.DETECT}`;
      const res = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Detection request failed");

      const data: DetectionResult = await res.json();
      setDetections(data);
    } catch (error) {
      console.error("Detection Error:", error);
      setDetections([]);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    detections,
    isLoading,
    detectItems,
    setDetections,
  };
};
