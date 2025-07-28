from dataclasses import dataclass
from io import BytesIO
from typing import Optional
import os
import pickle
import torch
from PIL import Image
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from trainer.datasets.base_dataset import BaseDataset, BaseDatasetConfig

logger = get_logger(__name__)


def simple_collate(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)


@dataclass
class ProcessorConfig:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")




@dataclass
class ICTHPDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasets.icthp_dataset.ICTHPDataset"
    dataset_name: str = ".Pick-High-Dataset/Pick-High/"
    dataset_config_name: str = "null"

    easy_folder: str = ".Pick-High-Dataset/pick_easy_img"
    refine_folder: str = ".Pick-High-Dataset/pick_refine_img" 
    

    from_disk: bool = False
    train_split_name: str = "train"
    valid_split_name: str = "val"
    test_split_name: str = "test"
    cache_dir: Optional[str] = None

    easy_caption_column_name: str = "easy_prompt"
    easy_image_0_column_name: str = "easy_image_0"
    easy_image_1_column_name: str = "easy_image_1"
    refine_caption_column_name: str = "refine_prompt" 
    refine_image_column_name: str = "refine_image"

    label_E1_column_name: str = "E1"
    label_E2_column_name: str = "E2"
    label_R1_column_name: str = "R1"
    label_R2_column_name: str = "R2"

    easy_input_ids_column_name: str = "easy_input_ids"
    refine_input_ids_column_name: str = "refine_input_ids"
    pixels_0_column_name: str = "pixel_values_0"
    pixels_1_column_name: str = "pixel_values_1"
    pixels_r_column_name: str = "pixel_values_r"

    keep_only_different: bool = False
    keep_only_with_label: bool = False
    keep_only_with_label_in_non_train: bool = True

    processor: ProcessorConfig = ProcessorConfig()

    limit_examples_per_prompt: int = -1

    only_on_best: bool = False




class ICTHPDataset(BaseDataset):

    def __init__(self, cfg: ICTHPDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        logger.info(f"Loading {self.split} dataset")

        self.dataset = self.load_by_dataset(self.split)
        logger.info(f"Loaded {len(self.dataset)} examples from {self.split} dataset")
        processor = instantiate(cfg.processor)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor


    def load_by_dataset(self, split: str) -> Dataset:
        dataset_path = os.path.join(self.cfg.dataset_name, split + ".pkl") 
        print("self.cfg.dataset_name",self.cfg.dataset_name)

        with open(dataset_path, 'rb') as file:
            dataset = pickle.load(file)
  
        return dataset


    def tokenize(self, example): 
        easy_caption = example[self.cfg.easy_caption_column_name]
        refine_caption = example[self.cfg.refine_caption_column_name]

        easy_input_ids = self.tokenizer(
            easy_caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        refine_input_ids = self.tokenizer(
            refine_caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return easy_input_ids, refine_input_ids

    def process_image(self, image_name, folder_path):
        image=Image.open(os.path.join(folder_path,self.split,image_name))
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError(f"Index {idx} is out of bounds for dataset of length {len(self.dataset)}")

        example = self.dataset.iloc[idx]

        easy_input_ids, refine_input_ids= self.tokenize(example)

        pixel_0_values = self.process_image(example[self.cfg.easy_image_0_column_name],self.cfg.easy_folder)
        pixel_1_values = self.process_image(example[self.cfg.easy_image_1_column_name],self.cfg.easy_folder)
        pixel_r_values = self.process_image(example[self.cfg.refine_image_column_name],self.cfg.refine_folder)

        item = {
            self.cfg.easy_input_ids_column_name: easy_input_ids,
            self.cfg.refine_input_ids_column_name: refine_input_ids,
            self.cfg.pixels_0_column_name: pixel_0_values,
            self.cfg.pixels_1_column_name: pixel_1_values,
            self.cfg.pixels_r_column_name: pixel_r_values,
            self.cfg.label_E1_column_name: torch.tensor(example[self.cfg.label_E1_column_name])[None],
            self.cfg.label_E2_column_name: torch.tensor(example[self.cfg.label_R1_column_name])[None],
            self.cfg.label_R1_column_name: torch.tensor(example[self.cfg.label_R1_column_name])[None],
            self.cfg.label_R2_column_name: torch.tensor(example[self.cfg.label_R2_column_name])[None],
        }
        return item

    def collate_fn(self, batch):
        easy_input_ids = simple_collate(batch, self.cfg.easy_input_ids_column_name)
        refine_input_ids = simple_collate(batch, self.cfg.refine_input_ids_column_name)
        pixel_0_values = simple_collate(batch, self.cfg.pixels_0_column_name)
        pixel_1_values = simple_collate(batch, self.cfg.pixels_1_column_name)
        pixel_r_values = simple_collate(batch, self.cfg.pixels_r_column_name)
        label_e1 = simple_collate(batch, self.cfg.label_E1_column_name)
        label_e2 = simple_collate(batch, self.cfg.label_E2_column_name)
        label_r1 = simple_collate(batch, self.cfg.label_R1_column_name)
        label_r2 = simple_collate(batch, self.cfg.label_R2_column_name)

        pixel_0_values = pixel_0_values.to(memory_format=torch.contiguous_format).float()
        pixel_1_values = pixel_1_values.to(memory_format=torch.contiguous_format).float()
        pixel_r_values = pixel_r_values.to(memory_format=torch.contiguous_format).float()

        collated = {
            self.cfg.easy_input_ids_column_name: easy_input_ids,
            self.cfg.refine_input_ids_column_name: refine_input_ids,
            self.cfg.pixels_0_column_name: pixel_0_values,
            self.cfg.pixels_1_column_name: pixel_1_values,
            self.cfg.pixels_r_column_name: pixel_r_values,
            self.cfg.label_E1_column_name: label_e1,
            self.cfg.label_E2_column_name: label_e2,
            self.cfg.label_R1_column_name: label_r1,
            self.cfg.label_R2_column_name: label_r2,
        }
        return collated

    def __len__(self):
        return len(self.dataset[self.cfg.easy_caption_column_name])
