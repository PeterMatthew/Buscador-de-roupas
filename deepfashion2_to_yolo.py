import pandas as pd
from PIL import Image
import shutil
import os
import yaml
from typing import Tuple
from dataclasses import dataclass
import json
import pandas as pd

class Deepfashion2DfBuilder:
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

    def load_annotations(self, image_id):
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.json")
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        return annotations

    def create_dataframe(self):
        data = []
        for image_filename in os.listdir(self.image_dir):
            image_id = os.path.splitext(image_filename)[0]
            annotations = self.load_annotations(image_id)

            for anno in annotations:
                if isinstance(annotations[anno], dict):
                    bounding_box = annotations[anno]['bounding_box']
                    category_name = annotations[anno]['category_name'].replace(' ', '_')
                    new_row = {'image_filename': image_filename, 'category_name': category_name, 'xmin': bounding_box[0],
                                'ymin': bounding_box[1], 'xmax': bounding_box[2], 'ymax': bounding_box[3],
                                'source': annotations['source'], 'pair_id': annotations['pair_id']}
                    data.append(new_row)

        return pd.DataFrame.from_dict(data)


@dataclass
class BoundingBox:
    top_left: Tuple[float, float]
    bottom_right: Tuple[float, float]


class YOLOAnnotationGenerator:
    def __init__(self, image_dir: str, target_path: str, categories: dict):
        self.image_dir = image_dir
        self.target_path = target_path
        self.categories = categories
        self.subsets = ['train', 'val', 'test']

    def generate_yolo_annotation(self, row: pd.DataFrame) -> str:
        bbox = BoundingBox((row['xmin'], row['ymin']), (row['xmax'], row['ymax']))

        with Image.open(os.path.join(self.image_dir, row['image_filename'])) as im:
            image_size = im.size

        x_center = (bbox.bottom_right[0] + bbox.top_left[0]) / 2
        y_center = (bbox.bottom_right[1] + bbox.top_left[1]) / 2
        width = bbox.bottom_right[0] - bbox.top_left[0]
        height = bbox.bottom_right[1] - bbox.top_left[1]

        return f"{self.categories[row['category_name']]} {x_center / image_size[0]} {y_center / image_size[1]} {width / image_size[0]} {height / image_size[1]}"

    def create_directories(self):
        if os.path.exists(self.target_path):
            shutil.rmtree(self.target_path)
        os.makedirs(self.target_path)
        for subset in self.subsets:
            for folder in ['images', 'labels']:
                os.makedirs(os.path.join(self.target_path, folder, subset))

    def copy_images_and_create_annotations(self, data: pd.DataFrame, subset_name: str):
        grouped_by_image = data.groupby("image_filename")
        
        for image_filename, image_group in grouped_by_image:
            yolo_annotations = image_group.apply(self.generate_yolo_annotation, axis=1)
            yolo_annotations = '\n'.join(yolo_annotations)
            
            annotation_path = os.path.join(self.target_path, "labels", subset_name, f"{image_filename.split('.')[0]}.txt")

            with open(annotation_path, "w") as file:
                file.write(yolo_annotations)

            shutil.copyfile(os.path.join(self.image_dir, image_filename),
                            os.path.join(self.target_path, "images", subset_name, image_filename))


def main():
    IMAGE_DIR = "deepfashion2/train/image"
    YOLO_DATASET_DIR = "yolo/datasets"
    YOLO_CONFIG_FILE = "yolo/config.yaml"
    EXPERIMENT_DIR = "experiment_3"

    with open(YOLO_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
        names = config.get('names', {})
        categories = {name: int(number) for number, name in names.items()}
    
    preprocessor = YOLOAnnotationGenerator(IMAGE_DIR, YOLO_DATASET_DIR, categories)
    preprocessor.create_directories()
    
    train = pd.read_csv(os.path.join("experiments", EXPERIMENT_DIR, "train_data.csv"))
    validation = pd.read_csv(os.path.join("experiments", EXPERIMENT_DIR, "validation_data.csv"))
    test = pd.read_csv(os.path.join("experiments", EXPERIMENT_DIR, "test_data.csv"))

    preprocessor.copy_images_and_create_annotations(train, "train")
    preprocessor.copy_images_and_create_annotations(validation, "val")
    preprocessor.copy_images_and_create_annotations(test, "test")
    
if __name__ == "__main__":
    main()
