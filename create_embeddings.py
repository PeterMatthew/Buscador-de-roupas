import torch
from tqdm import tqdm
from PIL import Image
from deepfashion2_to_yolo import Deepfashion2DfBuilder
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModel
from sklearn.neighbors import KDTree
import joblib
from torch.utils.data import DataLoader

INPUT_SIZE = 224
BATCH_SIZE = 256
IMAGE_DIR = "deepfashion2/train/image"
ANNOTATION_DIR = "deepfashion2/train/annos"
EMBEDDING_DIR = "embedding"

class DeepfashionDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms):
        self.dataframe = dataframe
        self.transforms = transforms

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        img = Image.open(os.path.join(IMAGE_DIR, row['image_filename'])).convert('RGB')

        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

        cropped_region = img.crop(bbox)
        
        if cropped_region.size[0] == 0 or cropped_region.size[1] == 0:
            return torch.zeros((3, 224, 224))

        img_tensor = self.transforms(cropped_region)
        return img_tensor

    def __len__(self):
        return len(self.dataframe)

class DinoTransform:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    model = model.to(device)
    model.eval()

    data_loader = Deepfashion2DfBuilder(IMAGE_DIR, ANNOTATION_DIR)

    dataset_df = data_loader.create_dataframe()
    dataset_df = dataset_df[dataset_df['source'] == 'shop']
    
    categories = dataset_df['category_name'].unique().tolist()

    custom_transform = DinoTransform(processor)
    
    for category in categories:
        print(f"solving category: {category}")
        category_df = dataset_df[dataset_df['category_name'] == category].reset_index(drop=True)

        dataset = DeepfashionDataset(category_df, transforms=custom_transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
        
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(loader):
                
                batch = batch.to(device)
                outputs = model(batch)
                
                emb = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(emb.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        embeddings_tree = KDTree(embeddings)

        if not os.path.exists(EMBEDDING_DIR):
            os.makedirs(EMBEDDING_DIR)
        tree_path = os.path.join(EMBEDDING_DIR, f"{category.replace(' ', '_')}_kdtree.pkl")
        joblib.dump(embeddings_tree, tree_path)
        
        np.savez(os.path.join(EMBEDDING_DIR, f"{category.replace(' ', '_')}_data.npz"),
             strings=category_df['image_filename'].to_numpy(),
             embeddings=embeddings)

        np.savez(os.path.join(EMBEDDING_DIR, f"{category.replace(' ', '_')}.npz"), embeddings=embeddings, strings=category_df['image_filename'].to_numpy())


if __name__ == "__main__":
    main()
