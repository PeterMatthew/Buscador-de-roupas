export interface Detection {
  box: number[];
  category: string;
}

export type DetectionResult = Detection[];

export interface PaginationInfo {
  limit: number;
  offset: number;
  total: number;
  next_offset: number | null;
  prev_offset: number | null;
}

export interface SimilarItem {
  image_name: string;
  image_url: string;
  score: number;
}

export interface SearchResponse {
  itemSummaries: SimilarItem[];
  pagination: PaginationInfo;
}
