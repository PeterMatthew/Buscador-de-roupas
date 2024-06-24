import torch
from tqdm import tqdm
from PIL import Image
from deepfashion2_to_yolo import Deepfashion2DfBuilder
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModel

INPUT_SIZE = 224
BATCH_SIZE = 256
IMAGE_DIR = "deepfashion2/train/image"
ANNOTATION_DIR = "deepfashion2/train/annos"
EMBEDDING_DIR = "embedding"

def roi_from_bbox(image, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    image = np.array(image)
    mask = np.zeros(image.shape,np.uint8)
    mask[y1:y2,x1:x2] = image[y1:y2,x1:x2]
    mask = Image.fromarray(mask)

    return mask

class DeepfashionDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms):
        self.dataframe = dataframe
        self.transforms = transforms

    def __getitem__(self, index):
        
        img = Image.open(os.path.join(IMAGE_DIR, self.dataframe['image_filename'][index]))

        bbox = (self.dataframe['xmin'][index], self.dataframe['ymin'][index], self.dataframe['xmax'][index], self.dataframe['ymax'][index])

        cropped_region = img.crop(bbox)
        cropped_region.thumbnail((768, 768))

        img = self.transforms(cropped_region)
        return img

    def __len__(self):
        return len(self.dataframe)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model = model.to(device)

    embedding_shape = 768

    data_loader = Deepfashion2DfBuilder(IMAGE_DIR, ANNOTATION_DIR)

    dataset_df = data_loader.create_dataframe()
    dataset_df = dataset_df[dataset_df['source'] == 'shop']
    
    categories = dataset_df['category_name'].unique().tolist()
    
    for category in categories:
        print(f"solving category: {category}")
        annotations = dataset_df[dataset_df['category_name'] == category].reset_index(drop=True)
        
        embeddings = np.zeros((len(annotations), embedding_shape))

        for i, row in tqdm(annotations.iterrows(), total=len(annotations)):
            img = Image.open(os.path.join(IMAGE_DIR, row['image_filename']))

            bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

            cropped_region = img.crop(bbox)
            if cropped_region.size[0] == 0 or cropped_region.size[1] == 0: continue

            with torch.no_grad():
                inputs = processor(images=cropped_region, return_tensors="pt").to(device)
                outputs = model(**inputs)
                image_features = outputs.last_hidden_state
                image_features = image_features.mean(dim=1)
                embeddings[i, :] = image_features.cpu().numpy()
        
        np.savez(os.path.join(EMBEDDING_DIR, f"{category.replace(' ', '_')}.npz"), embeddings=embeddings, strings=annotations['image_filename'].to_numpy())


if __name__ == "__main__":
    main()
