import collections
from dataclasses import dataclass

import torch
from PIL import Image
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from omegaconf import II
from transformers import AutoTokenizer

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask
from tqdm import tqdm

logger = get_logger(__name__)
import pdb

@dataclass
class ICTHPTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.icthp_task.ICTHPTask"
    pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    easy_input_ids_column_name: str = II("dataset.easy_input_ids_column_name")
    refine_input_ids_column_name: str = II("dataset.refine_input_ids_column_name")
    pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    pixels_1_column_name: str = II("dataset.pixels_1_column_name")
    pixels_r_column_name: str = II("dataset.pixels_r_column_name")

    label_E1_column_name: str = II("dataset.label_E1_column_name")
    label_R1_column_name: str = II("dataset.label_R1_column_name")
    label_R2_column_name: str = II("dataset.label_R2_column_name")
    batch_size: int = II("dataset.batch_size")



class ICTHPTask(BaseTask):
    def __init__(self, cfg: ICTHPTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
        self.cfg = cfg


    def train_step(self, model, criterion, batch, globel_step):
        loss ,ict_loss,icthp_loss = criterion(model, batch,globel_step)
        return loss ,ict_loss,icthp_loss

    @staticmethod
    def features2probs(model,image_0_features, image_1_features, image_r_features, text_features):
        image_0_scores =  torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_0_features))  
        image_1_scores = torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_1_features)) 
        image_r_scores =  torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_r_features)) 
        scores = torch.stack([image_0_scores, image_1_scores, image_r_scores], dim=-1)

        return scores 

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        image_0_features, image_1_features, image_r_features, easy_text_features,hp_0_scores, hp_1_scores,hp_r_scores = criterion.get_features(
            model,
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name],
            batch[self.cfg.pixels_r_column_name],
            batch[self.cfg.easy_input_ids_column_name],
            is_hp = True
        )
        ict_probs = self.features2probs(model, image_0_features, image_1_features, image_r_features, easy_text_features)
        hp_scores = torch.cat([hp_0_scores, hp_1_scores,hp_r_scores], dim=1)
        hp_scores = torch.softmax(hp_scores, dim=-1)

        return ict_probs,hp_scores 


    def run_inference(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        logger.info("Running ICT-HP accuracy...")

        for batch in tqdm(dataloader, total=len(dataloader), desc="Processing Batches"):

            ict_probs,hp_scores  = self.valid_step(model, criterion, batch)  
            ict_hp_scores = ict_probs * hp_scores
            hp_0_scores,hp_1_scores,hp_r_scores = hp_scores.chunk(3, dim=-1) 
            ict_0_scores,ict_1_scores,ict_r_scores = ict_hp_scores.chunk(3, dim=-1) 
      
            agree_on_10 = (ict_1_scores > ict_0_scores).T *  torch.ones(len(ict_0_scores), device=ict_0_scores.device)
            agree_on_r0 = (ict_r_scores > ict_0_scores).T *  torch.ones(len(ict_0_scores), device=ict_0_scores.device)
            agree_on_r1 = (ict_r_scores > ict_1_scores).T *  torch.ones(len(ict_0_scores), device=ict_0_scores.device)
            is_correct = agree_on_10[0] + agree_on_r0[0] + agree_on_r1[0]
            eval_dict["is_correct"] += is_correct.tolist()
            eval_dict["agree_on_10"] += agree_on_10[0].tolist()
            eval_dict["agree_on_r0"] += agree_on_r0[0].tolist()
            eval_dict["agree_on_r1"] += agree_on_r1[0].tolist()


            hp_10 = (hp_1_scores > hp_0_scores).T *  torch.ones(len(hp_0_scores), device=hp_0_scores.device)
            hp_r0 = (hp_r_scores > hp_0_scores).T *  torch.ones(len(hp_0_scores), device=hp_0_scores.device)
            hp_r1 = (hp_r_scores > hp_1_scores).T *  torch.ones(len(hp_0_scores), device=hp_0_scores.device)
            hp_correct = hp_10[0] + hp_r0[0] + hp_r1[0]
            eval_dict["hp_correct"] += hp_correct.tolist()
            eval_dict["hp_10"] += hp_10[0].tolist()
            eval_dict["hp_r0"] += hp_r0[0].tolist()
            eval_dict["hp_r1"] += hp_r1[0].tolist()
            
            eval_dict['hp_0'] += hp_0_scores.T[0].tolist()
            eval_dict['hp_1'] += hp_1_scores.T[0].tolist()
            eval_dict['hp_r'] += hp_r_scores.T[0].tolist()

        return eval_dict

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_inference(model, criterion, dataloader) 
        eval_dict = self.gather_dict(eval_dict)
        metrics = {
            "accuracy": sum(eval_dict["is_correct"]) / (len(dataloader.dataset) * 3), 
            "num_samples": (len(dataloader.dataset) * 3), 
            "agree_on_10":sum(eval_dict["agree_on_10"]) / (len(dataloader.dataset)),
            "agree_on_r0":sum(eval_dict["agree_on_r0"]) / (len(dataloader.dataset)),
            "agree_on_r1":sum(eval_dict["agree_on_r1"]) / (len(dataloader.dataset)),

            "hp_acc:": sum(eval_dict["hp_correct"]) / (len(dataloader.dataset) * 3), 
            "hp_10" : sum(eval_dict["hp_10"]) / (len(dataloader.dataset)),
            "hp_r0" : sum(eval_dict["hp_r0"]) / (len(dataloader.dataset)),
            "hp_r1" : sum(eval_dict["hp_r1"]) / (len(dataloader.dataset)),

            "hp_0": sum(eval_dict['hp_0'])/  (len(dataloader.dataset)),
            "hp_1": sum(eval_dict['hp_1'])/  (len(dataloader.dataset)),
            "hp_r": sum(eval_dict['hp_r'])/  (len(dataloader.dataset)),
        }

        return metrics
