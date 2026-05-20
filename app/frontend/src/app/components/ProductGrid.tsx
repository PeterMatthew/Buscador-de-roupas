import { SimilarItem, PaginationInfo } from "../types";

interface ProductGridProps {
  items: SimilarItem[];
  pagination: PaginationInfo | null;
  onNextPage: () => void;
  onPrevPage: () => void;
}

export const ProductGrid: React.FC<ProductGridProps> = ({
  items,
  pagination,
  onNextPage,
  onPrevPage,
}) => {
  return (
    <div className="w-full">
      {/* Pagination */}
      <div className="flex justify-between items-center mb-4">
        <div className="text-sm">
          Showing {pagination ? pagination.offset + 1 : 0} -{" "}
          {pagination
            ? Math.min(pagination.offset + pagination.limit, pagination.total)
            : 0}{" "}
          of {pagination?.total || 0}
        </div>
        <div className="flex gap-2">
          <button
            onClick={onPrevPage}
            disabled={
              !pagination ||
              pagination.prev_offset === null ||
              pagination.offset === 0
            }
            className="border px-3 py-1 text-sm disabled:opacity-50 cursor-pointer hover:bg-gray-50"
          >
            Previous
          </button>
          <button
            onClick={onNextPage}
            disabled={!pagination || pagination.next_offset === null}
            className="border px-3 py-1 text-sm disabled:opacity-50 cursor-pointer hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      </div>

      {/* Products */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {items.map((item, idx) => (
          <div
            key={idx}
            className="block border hover:border-black transition-colors"
          >
            <div className="aspect-square overflow-hidden bg-gray-100">
              <img
                src={item.image_url}
                alt={item.image_name}
                className="w-full h-full object-contain p-2"
              />
            </div>
            <div className="p-3">
              <p className="text-xs text-gray-500 mb-1">{item.image_name}</p>
              <p className="text-sm font-medium">
                Similarity: {(item.score * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
