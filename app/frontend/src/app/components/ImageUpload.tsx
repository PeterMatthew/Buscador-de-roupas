interface ImageUploadProps {
  onUpload: (file: File) => void;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({ onUpload }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="text-center mb-8">
      <label className="cursor-pointer inline-block border border-black px-6 py-3 hover:bg-black hover:text-white">
        <span>Upload Image</span>
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          className="hidden"
        />
      </label>
    </div>
  );
};
