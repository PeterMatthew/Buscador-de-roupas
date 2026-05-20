import { useState } from "react";
import {
  Detection,
  SearchResponse,
  SimilarItem,
  PaginationInfo,
} from "../types";
import { API_CONFIG } from "../config";

export const useSearch = () => {
  const [similarItems, setSimilarItems] = useState<SimilarItem[]>([]);
  const [pagination, setPagination] = useState<PaginationInfo | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const search = async (imageFile: File, detection: Detection, offset = 0) => {
    setIsSearching(true);
    const [x0, y0, x1, y1] = detection.box;

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("x0", x0.toString());
    formData.append("y0", y0.toString());
    formData.append("x1", x1.toString());
    formData.append("y1", y1.toString());
    formData.append("category", detection.category);
    formData.append("offset", offset.toString());

    try {
      const res = await fetch(
        `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.SIMILAR}`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data: SearchResponse = await res.json();
      setSimilarItems(data.itemSummaries);
      setPagination(data.pagination);
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setIsSearching(false);
    }
  };

  return {
    similarItems,
    pagination,
    isSearching,
    search,
    setSimilarItems,
    setPagination,
  };
};
