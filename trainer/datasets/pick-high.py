import pickle
import pandas as pd
from PIL import Image
import os

# Load structured data
def load_split_data(split='train'):
    with open(f'./Pick-High-Dataset/Pick-High/{split}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# Load images
def load_image(image_filename, image_type='easy', split='train'):
    folder = f'./Pick-High-Dataset/pick_{image_type}_img/{split}'
    print(folder)
    image_path = os.path.join(folder, image_filename)
    return Image.open(image_path)

# Example usage
train_data = load_split_data('train')
print(f"Training samples: {len(train_data)}")

# Access a specific record
sample = train_data.iloc[0]
print(sample.keys())
easy_img = load_image(sample['easy_image_0'], 'easy', 'train')
refined_img = load_image(sample['refine_image'], 'refine', 'train')