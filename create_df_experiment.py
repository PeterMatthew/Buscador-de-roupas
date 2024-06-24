import os
import numpy as np
import pandas as pd
from deepfashion2_to_yolo import Deepfashion2DfBuilder

def split_dataset(df, train_size, validation_size, test_size):
    fractions = np.array([train_size, validation_size, test_size])
    
    df = df.sample(frac=1) # shuffle

    train, validation, test = np.array_split(
        df, (fractions[:-1].cumsum() * len(df)).astype(int))
    return train, validation, test

def save_dataframe(df, directory, filename):
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)

def main():
    IMAGE_DIR = "deepfashion2/train/image"
    ANNOTATION_DIR = "deepfashion2/train/annos"

    data_builder = Deepfashion2DfBuilder(IMAGE_DIR, ANNOTATION_DIR)
    data = data_builder.create_dataframe()
    

    # experimento 1: 250 objetos de cada categoria
    N_SAMPLES = 250
    TRAIN_FRACTION = 0.7
    VALIDATION_FRACTION = 0.15
    TEST_FRACTION = 0.15

    sampled_df = data.groupby('category_name').apply(lambda s: s.sample(min(len(s), N_SAMPLES)), include_groups=False).reset_index()

    train_df, validation_df, test_df = [], [], []

    for category in sampled_df['category_name'].unique():
        category_df = sampled_df[sampled_df['category_name'] == category]
        train, validation, test = split_dataset(category_df, TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION)
        train_df.append(train)
        validation_df.append(validation)
        test_df.append(test)

    train = pd.concat(train_df)
    validation = pd.concat(validation_df)
    test = pd.concat(test_df)

    train = data[data['image_filename'].isin(train['image_filename'])]
    validation = data[data['image_filename'].isin(validation['image_filename'])]
    test = data[data['image_filename'].isin(test['image_filename'])]

    train = train[~train['image_filename'].isin(validation['image_filename']) & ~train['image_filename'].isin(test['image_filename'])]
    validation = validation[~validation['image_filename'].isin(train['image_filename']) & ~validation['image_filename'].isin(test['image_filename'])]
    test = test[~test['image_filename'].isin(train['image_filename']) & ~test['image_filename'].isin(validation['image_filename'])]

    experiment_1_dir = os.path.join("experiments", "experiment_1")
    os.makedirs(experiment_1_dir, exist_ok=True)
    save_dataframe(train, experiment_1_dir, "train_data.csv")
    save_dataframe(validation, experiment_1_dir, "validation_data.csv")
    save_dataframe(test, experiment_1_dir, "test_data.csv")

    # experimento 2: 500 objetos de cada categoria
    N_SAMPLES = 500

    sampled_df = data.groupby('category_name').apply(lambda s: s.sample(min(len(s), N_SAMPLES)), include_groups=False).reset_index()

    train_df, validation_df, test_df = [], [], []

    for category in sampled_df['category_name'].unique():
        category_df = sampled_df[sampled_df['category_name'] == category]
        train, validation, test = split_dataset(category_df, TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION)
        train_df.append(train)
        validation_df.append(validation)
        test_df.append(test)

    train = pd.concat(train_df)
    validation = pd.concat(validation_df)
    test = pd.concat(test_df)

    train = data[data['image_filename'].isin(train['image_filename'])]
    validation = data[data['image_filename'].isin(validation['image_filename'])]
    test = data[data['image_filename'].isin(test['image_filename'])]

    train = train[~train['image_filename'].isin(validation['image_filename']) & ~train['image_filename'].isin(test['image_filename'])]
    validation = validation[~validation['image_filename'].isin(train['image_filename']) & ~validation['image_filename'].isin(test['image_filename'])]
    test = test[~test['image_filename'].isin(train['image_filename']) & ~test['image_filename'].isin(validation['image_filename'])]

    experiment_2_dir = os.path.join("experiments", "experiment_2")
    os.makedirs(experiment_2_dir, exist_ok=True)
    save_dataframe(train, experiment_2_dir, "train_data.csv")
    save_dataframe(validation, experiment_2_dir, "validation_data.csv")
    save_dataframe(test, experiment_2_dir, "test_data.csv")

    # experimento 3: 1000 objetos de cada categoria
    N_SAMPLES = 1000

    sampled_df = data.groupby('category_name').apply(lambda s: s.sample(min(len(s), N_SAMPLES)), include_groups=False).reset_index()

    train_df, validation_df, test_df = [], [], []

    for category in sampled_df['category_name'].unique():
        category_df = sampled_df[sampled_df['category_name'] == category]
        train, validation, test = split_dataset(category_df, TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION)
        train_df.append(train)
        validation_df.append(validation)
        test_df.append(test)

    train = pd.concat(train_df)
    validation = pd.concat(validation_df)
    test = pd.concat(test_df)

    train = data[data['image_filename'].isin(train['image_filename'])]
    validation = data[data['image_filename'].isin(validation['image_filename'])]
    test = data[data['image_filename'].isin(test['image_filename'])]

    train = train[~train['image_filename'].isin(validation['image_filename']) & ~train['image_filename'].isin(test['image_filename'])]
    validation = validation[~validation['image_filename'].isin(train['image_filename']) & ~validation['image_filename'].isin(test['image_filename'])]
    test = test[~test['image_filename'].isin(train['image_filename']) & ~test['image_filename'].isin(validation['image_filename'])]

    experiment_3_dir = os.path.join("experiments", "experiment_3")
    os.makedirs(experiment_3_dir, exist_ok=True)
    save_dataframe(train, experiment_3_dir, "train_data.csv")
    save_dataframe(validation, experiment_3_dir, "validation_data.csv")
    save_dataframe(test, experiment_3_dir, "test_data.csv")

if __name__ == "__main__":
    main()
