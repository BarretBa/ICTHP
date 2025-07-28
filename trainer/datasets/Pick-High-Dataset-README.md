# Pick-High-Dataset
## Overview
**Pick-High-Dataset** is a large-scale, high-quality dataset designed for training and evaluating reward models for image generation. The dataset contains 360,000 prompts from the PickAPic v2 dataset and leveraged large language models' chain-of-thought capabilities to meticulously design a more refined set of prompts that closely align with human preferences, creating 360,000 high-quality images.

### Key Features
- **360,000 high-quality image triplets** with preference rankings
- **Refined prompts** generated using LLM chain-of-thought reasoning
- **Hierarchical ICT labels** (E1, E2, R1, R2) for comprehensive quality assessment
- **Beyond text-image alignment** - captures aesthetic quality and visual richness

## Dataset Structure
```
Pick-High-Dataset/
├── Pick-High/
│   ├── train.pkl          # Training data with structured annotations
│   ├── val.pkl           # Validation data
│   └── test.pkl          # Test data
├── pick_easy_img/
│   ├── train/            # Original images from Pick-a-pic dataset
│   ├── val/
│   └── test/
└── pick_refine_img/
    ├── train/            # Newly generated images using refined prompts
    ├── val/
    └── test/
```

## Data Format
Each record contains:

**Text Fields**
- `easy_prompt`: Original prompt from Pick-a-pic dataset
- `refine_prompt`: Enhanced prompt generated using chain-of-thought reasoning

**Image Fields**
- `easy_image_0`: Lose image from Pick-a-pic dataset
- `easy_image_1`: Win image from Pick-a-pic dataset
- `refine_image`: Newly generated high-quality image created from refined prompt using Stable-Diffusion-3.5-Large

**ICT Labels**
- `E1`: Basic prompt ICT score for easy_image_0 (lose image)
- `E2`: Basic prompt ICT score for easy_image_1 (win image)
- `R1`: Refined prompt ICT score for easy_image_0
- `R2`: Refined prompt ICT score for easy_image_1

## Usage
### Loading the Dataset
```python
import pickle
import pandas as pd
from PIL import Image
import os

def load_split_data(split='train', dataset_path='Pick-High-Dataset'):
    """
    Load structured data from pickle files.
    
    Args:
        split (str): Data split ('train', 'val', 'test')
        dataset_path (str): Path to the dataset directory
    
    Returns:
        pd.DataFrame: Loaded data with annotations
    """
    file_path = os.path.join(dataset_path, 'Pick-High', f'{split}.pkl')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_image(image_filename, image_type='easy', split='train', dataset_path='Pick-High-Dataset'):
    """
    Load image from the dataset.
    
    Args:
        image_filename (str): Name of the image file
        image_type (str): Type of image ('easy' or 'refine')
        split (str): Data split ('train', 'val', 'test')
        dataset_path (str): Path to the dataset directory
    
    Returns:
        PIL.Image: Loaded image
    """
    folder = os.path.join(dataset_path, f'pick_{image_type}_img', split)
    image_path = os.path.join(folder, image_filename)
    return Image.open(image_path)

# Example usage
dataset_path = 'path/to/Pick-High-Dataset'  # Update with your dataset path

# Load training data
train_data = load_split_data('train', dataset_path)
print(f"Training samples: {len(train_data)}")

# Access a specific record
sample = train_data.iloc[0]

# Load corresponding images
easy_img_0 = load_image(sample['easy_image_0'], 'easy', 'train', dataset_path)  # Pick-a-pic lose image
easy_img_1 = load_image(sample['easy_image_1'], 'easy', 'train', dataset_path)  # Pick-a-pic win image
refined_img = load_image(sample['refine_image'], 'refine', 'train', dataset_path)  # Newly generated image
```

## Data Collection
1. **Original Data**: 360,000 prompts and corresponding win/lose image pairs from Pick-a-pic dataset
2. **Image Organization**: 
   - `pick_easy_img/`: Contains original images from Pick-a-pic dataset, organized with "_0" suffix for lose images and "_1" suffix for win images
   - `pick_refine_img/`: Contains newly generated high-quality images created using refined prompts
3. **Prompt Refinement**: Original Pick-a-pic prompts enhanced using GPT-2 PromptExtend and Claude-3.5-Sonnet with chain-of-thought reasoning
4. **Image Generation**: New high-quality images generated from refined prompts using Stable Diffusion-3.5-Large
5. **Quality Control**: Expert evaluation confirmed 97% prompt and 95% image compliance


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources
- **🤗 ICT Model**: [8y/ICT - Text-Image Alignment Model](https://huggingface.co/8y/ICT)
- **🤗 HP Model**: [8y/HP - Aesthetic Quality Model](https://huggingface.co/8y/HP)
- **📊 Dataset**: [Pick-High Dataset on Hugging Face](https://huggingface.co/datasets/8y/Pick-High-Dataset)
- **📄 Paper**: [Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](https://arxiv.org/abs/2507.19002)
- **🔗 Base Project**: [PickScore - Pick-a-Pic Dataset and PickScore Model](https://github.com/yuvalkirstain/PickScore)
