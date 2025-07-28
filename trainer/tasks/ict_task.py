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

@dataclass
class ICTTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.ict_task.ICTTask"
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

class ICTTask(BaseTask):
    def __init__(self, cfg: ICTTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
        self.cfg = cfg

    def train_step(self, model, criterion, batch, globel_step):
        loss, ict_loss, icthp_loss = criterion(model, batch, globel_step)
        return loss, ict_loss, icthp_loss

    @staticmethod
    def features2probs(model, image_0_features, image_1_features, image_r_features, text_features, refine_text_features=None):
        if refine_text_features is not None:
            image_r0_scores = torch.diag(
                torch.einsum('bd,cd->bc', refine_text_features, image_0_features))
            image_r1_scores = torch.diag(
                torch.einsum('bd,cd->bc', refine_text_features, image_1_features))
            image_rr_scores = torch.diag(
                torch.einsum('bd,cd->bc', refine_text_features, image_r_features))
            r_scores = torch.stack([image_r0_scores, image_r1_scores, image_rr_scores], dim=-1)

        image_e0_scores = torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_e1_scores = torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_1_features))
        image_er_scores = torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_r_features))
        e_scores = torch.stack([image_e0_scores, image_e1_scores, image_er_scores], dim=-1)

        if refine_text_features is None:
            return e_scores
        else:
            return e_scores, r_scores

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        image_0_features, image_1_features, image_r_features, easy_text_features, refine_text_features, hp_0_scores, hp_1_scores, hp_r_scores = criterion.get_features(
            model,
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name],
            batch[self.cfg.pixels_r_column_name],
            batch[self.cfg.easy_input_ids_column_name],
            batch[self.cfg.refine_input_ids_column_name],
            is_hp=True
        )
        e_ict_probs, r_ict_probs = self.features2probs(model, image_0_features, image_1_features, image_r_features, easy_text_features, refine_text_features)
        hp_scores = torch.cat([hp_0_scores, hp_1_scores, hp_r_scores], dim=1)
        hp_scores = torch.softmax(hp_scores, dim=-1)

        return e_ict_probs, r_ict_probs

    def run_inference(self, model, criterion, dataloader):
        eval_dict = collections.defaultdict(list)
        logger.info("Running ICT score...")

        for batch in tqdm(dataloader, total=len(dataloader), desc="Processing Batches"):
            e_ict_probs, r_ict_probs = self.valid_step(model, criterion, batch)
            ict_e0, ict_e1, ict_er = e_ict_probs.chunk(3, dim=-1)
            ict_r0, ict_r1, ict_rr = r_ict_probs.chunk(3, dim=-1)

            e1 = batch['E1'].to(ict_e0.device)
            e2 = torch.ones(e1.shape[0], device=ict_e0.device, dtype=torch.float16)
            e3 = torch.ones(e1.shape[0], device=ict_e0.device, dtype=torch.float16)

            e_ict_10 = (ict_e1 > ict_e0).T * torch.ones(len(ict_e0), device=ict_e0.device)
            e_ict_r0 = (ict_er > ict_e0).T * torch.ones(len(ict_e0), device=ict_e0.device)
            e_ict_r1 = (ict_er > ict_e1).T * torch.ones(len(ict_e0), device=ict_e0.device)
            ict_e_correct = e_ict_10[0] + e_ict_r0[0] + e_ict_r1[0]
            eval_dict["ict_e_correct"] += ict_e_correct.tolist()

            losse0 = (e1 - ict_e0.T[0]) ** 2
            losse1 = (e2 - ict_e1.T[0]) ** 2
            losse2 = (e3 - ict_er.T[0]) ** 2
            total_e_loss = losse0 + losse1 + losse2
            eval_dict['total_e_loss'] += total_e_loss.tolist()

            r_ict_10 = (ict_r1 > ict_r0).T * torch.ones(len(ict_e0), device=ict_e0.device)
            r_ict_r0 = (ict_rr > ict_r0).T * torch.ones(len(ict_e0), device=ict_e0.device)
            r_ict_r1 = (ict_rr > ict_r1).T * torch.ones(len(ict_e0), device=ict_e0.device)
            ict_r_correct = r_ict_10[0] + r_ict_r0[0] + r_ict_r1[0]
            eval_dict["ict_r_correct"] += ict_r_correct.tolist()

            lossr0 = (e1 - ict_r0.T[0]) ** 2
            lossr1 = (e2 - ict_r1.T[0]) ** 2
            lossr2 = (e3 - ict_rr.T[0]) ** 2
            total_r_loss = lossr0 + lossr1 + lossr2
            eval_dict['total_r_loss'] += total_r_loss.tolist()

            eval_dict["ict_correct"] += ict_e_correct.tolist() + ict_r_correct.tolist()
            eval_dict['total_loss'] += total_e_loss.tolist() + total_r_loss.tolist()

        return eval_dict

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader):
        eval_dict = self.run_inference(model, criterion, dataloader)
        eval_dict = self.gather_dict(eval_dict)
        metrics = {
            "ict_acc": sum(eval_dict["ict_correct"]) / (len(dataloader.dataset) * 6),
            "total_loss": sum(eval_dict['total_loss']) / (len(dataloader.dataset) * 6),
            "ict_e_acc": sum(eval_dict["ict_e_correct"]) / (len(dataloader.dataset) * 3),
            "ict_r_acc": sum(eval_dict["ict_r_correct"]) / (len(dataloader.dataset) * 3),
            "total_e_loss": sum(eval_dict["total_e_loss"]) / (len(dataloader.dataset) * 3),
            "total_r_loss": sum(eval_dict["total_r_loss"]) / (len(dataloader.dataset) * 3),
        }

        return metrics
