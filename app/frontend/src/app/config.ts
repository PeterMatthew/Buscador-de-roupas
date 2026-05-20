export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL,
  ENDPOINTS: {
    DETECT: "/detect",
    SIMILAR: "/similar",
  },
} as const;
