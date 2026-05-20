import React from "react";

interface Props {
  bbox: number[];
  imageWidth: number;
  imageHeight: number;
  category: string;
  onClick: () => void;
  isSelected: boolean;
}

const categoryColors: { [key: string]: string } = {
  short_sleeve_top: "#FF0000",
  long_sleeve_top: "#00FF00",
  skirt: "#0000FF",
  shorts: "#FFFF00",
  trousers: "#FF00FF",
  short_sleeve_dress: "#00FFFF",
  long_sleeve_dress: "#FFA500",
  vest_dress: "#800080",
  sling_dress: "#008000",
  sling: "#FF4500",
  vest: "#4B0082",
  short_sleeve_outwear: "#DC143C",
  long_sleeve_outwear: "#2E8B57",
};

const DetectedItem = ({
  bbox,
  imageWidth,
  imageHeight,
  category,
  onClick,
  isSelected,
}: Props) => {
  const [xmin, ymin, xmax, ymax] = bbox;

  const centerX = ((xmin + xmax) / 2 / imageWidth) * 100;
  const centerY = ((ymin + ymax) / 2 / imageHeight) * 100;

  const bboxColor = categoryColors[category] || "#808080";

  const buttonStyle = {
    position: "absolute" as "absolute",
    left: `${centerX}%`,
    top: `${centerY}%`,
    transform: "translate(-50%, -50%)",
  };

  const bboxStyle = {
    position: "absolute" as "absolute",
    left: `${(xmin / imageWidth) * 100}%`,
    top: `${(ymin / imageHeight) * 100}%`,
    width: `${((xmax - xmin) / imageWidth) * 100}%`,
    height: `${((ymax - ymin) / imageHeight) * 100}%`,
    border: `2px solid ${bboxColor}`,
    backgroundColor: "transparent",
    pointerEvents: "none" as "none",
    borderRadius: "2px",
  };

  return (
    <>
      {/* Bounding Box */}
      <div style={bboxStyle} />

      {/* Button */}
      <div style={buttonStyle} className="z-10">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onClick();
          }}
          className={`
            px-3 py-1 text-xs font-medium uppercase tracking-wide border cursor-pointer
            ${
              isSelected
                ? "bg-black text-white border-black"
                : "bg-white text-black border-gray-600 hover:border-black"
            }
          `}
          style={{
            borderColor: bboxColor,
            backgroundColor: isSelected ? bboxColor : "white",
            color: isSelected ? "white" : bboxColor,
            fontWeight: "bold",
          }}
        >
          {category.replaceAll("_", " ")}
        </button>
      </div>
    </>
  );
};

export default DetectedItem;
