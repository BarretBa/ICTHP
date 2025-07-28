from dataclasses import dataclass
import torch
from omegaconf import II
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn

@dataclass
class ICTHPCriterionConfig:
    _target_: str = "trainer.criterions.icthp_criterion.ICTHPCriterion"
    is_distributed: bool = True
    easy_input_ids_column_name: str = II("dataset.easy_input_ids_column_name")
    refine_input_ids_column_name: str = II("dataset.refine_input_ids_column_name")
    pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    pixels_1_column_name: str = II("dataset.pixels_1_column_name")
    pixels_r_column_name: str = II("dataset.pixels_r_column_name")
    label_E1_column_name: str = II("dataset.label_E1_column_name")
    label_E2_column_name: str = II("dataset.label_E2_column_name")
    label_R1_column_name: str = II("dataset.label_R1_column_name")
    label_R2_column_name: str = II("dataset.label_R2_column_name")
    loss_ict: bool = True
    loss_hp: bool = False
    loss_icthp: bool = False
    ict_weight: float = 1.0
    hp_weight: float = 1.0
    icthp_weight: float = 1.0
    neg_weight: float = 0.5
    margin: float = 0.3

class ICTHPCriterion(_Loss):
    def __init__(self, cfg: ICTHPCriterionConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_features(model, pixels_0_values, pixels_1_values, pixel_r_values, easy_input_ids, refine_input_ids=None, is_hp=None):
        all_pixel_values = torch.cat([pixels_0_values, pixels_1_values, pixel_r_values], dim=0)
        all_text_ids = torch.cat([easy_input_ids, refine_input_ids], dim=0) if refine_input_ids is not None else easy_input_ids
        
        if is_hp is not None:
            all_text_features, all_image_features, all_hp_scores = model(
                text_inputs=all_text_ids, image_inputs=all_pixel_values, is_hp=is_hp
            )
        else:
            all_text_features, all_image_features = model(
                text_inputs=all_text_ids, image_inputs=all_pixel_values, is_hp=is_hp
            )
        
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
        image_0_features, image_1_features, image_r_features = all_image_features.chunk(3, dim=0)
        
        if is_hp is not None:
            hp_0_scores, hp_1_scores, hp_r_scores = all_hp_scores.chunk(3, dim=0)
            if refine_input_ids is not None:
                easy_text_features, refine_text_features = all_text_features.chunk(2, dim=0)
                return image_0_features, image_1_features, image_r_features, easy_text_features, refine_text_features, hp_0_scores, hp_1_scores, hp_r_scores
            else:
                easy_text_features = all_text_features
                return image_0_features, image_1_features, image_r_features, easy_text_features, hp_0_scores, hp_1_scores, hp_r_scores
        else:
            if refine_input_ids is not None:
                easy_text_features, refine_text_features = all_text_features.chunk(2, dim=0)
                return image_0_features, image_1_features, image_r_features, easy_text_features, refine_text_features
            else:
                easy_text_features = all_text_features
                return image_0_features, image_1_features, image_r_features, easy_text_features

    @staticmethod
    def gather_features(features):
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features
    
    def get_negative_logits(self, text_0_logits, text_1_logits, text_r_logits):
        batch_size = text_0_logits.shape[0]
        diag_mask = torch.eye(batch_size, device=text_0_logits.device, dtype=torch.bool)
        negative_mask = ~diag_mask

        negative_0_logits = text_0_logits[negative_mask]
        negative_1_logits = text_1_logits[negative_mask]
        negative_r_logits = text_r_logits[negative_mask]

        negative_logits = torch.stack([negative_0_logits, negative_1_logits, negative_r_logits], dim=-1).to(dtype=torch.bfloat16)
        return negative_logits
    
    def compute_weights_smooth(self, data, threshold=6, sharpness=20):
        abs_data = torch.abs(data)
        weights = 1 / (1 + torch.exp(sharpness * (abs_data - threshold)))
        return weights

    def calc_loss(self, easy_text_features, refine_text_features, image_0_features, image_1_features, image_r_features,
                  hp_0_scores, hp_1_scores, hp_r_scores, globel_step, label_e1, label_e2, label_r1, label_r2, *args, **kwargs):
        device = image_0_features.device

        if self.cfg.is_distributed:
            image_0_features = self.gather_features(image_0_features)
            image_1_features = self.gather_features(image_1_features)
            image_r_features = self.gather_features(image_r_features)
            hp_0_scores = self.gather_features(hp_0_scores)
            hp_1_scores = self.gather_features(hp_1_scores)
            hp_r_scores = self.gather_features(hp_r_scores)
            easy_text_features = self.gather_features(easy_text_features)
            refine_text_features = self.gather_features(refine_text_features)
            label_e1 = self.gather_features(label_e1)
            label_e2 = self.gather_features(label_e2)
            label_r1 = self.gather_features(label_r1)
            label_r2 = self.gather_features(label_r2)

        all_image_features = torch.cat([image_0_features, image_1_features, image_r_features], dim=0)
        logits_per_image = 90 * all_image_features @ easy_text_features.T
        image_0_logits, image_1_logits, image_r_logits = logits_per_image.chunk(3, dim=0)

        e_text_logits = easy_text_features @ all_image_features.T
        r_text_logits = refine_text_features @ all_image_features.T

        text_e0_logits, text_e1_logits, text_er_logits = e_text_logits.chunk(3, dim=-1)
        text_r0_logits, text_r1_logits, text_rr_logits = r_text_logits.chunk(3, dim=-1)
        
        negative_e_text_logits = self.get_negative_logits(text_e0_logits, text_e1_logits, text_er_logits)
        negative_r_text_logits = self.get_negative_logits(text_r0_logits, text_r1_logits, text_rr_logits)

        index = torch.arange(text_e0_logits.shape[0], device=device, dtype=torch.long)
        text_e0_logits = text_e0_logits[index, index]
        text_e1_logits = text_e1_logits[index, index]
        text_er_logits = text_er_logits[index, index]
        text_r0_logits = text_r0_logits[index, index]
        text_r1_logits = text_r1_logits[index, index]
        text_rr_logits = text_rr_logits[index, index]

        e_text_logits = torch.stack([text_e0_logits, text_e1_logits, text_er_logits], dim=-1).to(dtype=torch.bfloat16)
        r_text_logits = torch.stack([text_r0_logits, text_r1_logits, text_rr_logits], dim=-1).to(dtype=torch.bfloat16)
        
        label_one = torch.ones(e_text_logits.shape[0], device=device, dtype=torch.float16)
        e_labels = torch.stack([label_e1, label_e2, label_one], dim=-1).to(dtype=torch.bfloat16)
        r_labels = torch.stack([label_r1, label_r2, label_one], dim=-1).to(dtype=torch.bfloat16)

        label_zero = torch.zeros(negative_e_text_logits.shape[0], device=device, dtype=torch.float16)
        label_zeros = torch.stack([label_zero, label_zero, label_zero], dim=-1).to(dtype=torch.bfloat16)

        criterion = nn.MSELoss(reduction='none')
        
        if self.cfg.loss_ict:
            text_e_loss = criterion(e_text_logits, e_labels)
            text_r_loss = criterion(r_text_logits, r_labels)
            neg_e_loss = criterion(negative_e_text_logits, label_zeros)
            neg_r_loss = criterion(negative_r_text_logits, label_zeros)
            
            weights_e = self.compute_weights_smooth(negative_e_text_logits)
            weights_r = self.compute_weights_smooth(negative_r_text_logits)
            
            neg_e_loss *= weights_e
            neg_r_loss *= weights_r
            
            ict_loss = text_e_loss.mean() + text_r_loss.mean() + \
                      self.cfg.neg_weight * (neg_e_loss.mean() + neg_r_loss.mean())

        all_hp_scores = torch.cat([hp_0_scores, hp_1_scores, hp_r_scores], dim=1)
        all_hp_scores = torch.softmax(all_hp_scores, dim=-1)
        ict_hp_scores = e_text_logits * all_hp_scores
        icthp_0_scores, icthp_1_scores, icthp_r_scores = ict_hp_scores.chunk(3, dim=1)

        target_one = torch.ones(hp_0_scores.shape, device=device)
        
        hp_10_loss = F.margin_ranking_loss(hp_1_scores, hp_0_scores, target_one, margin=self.cfg.margin)
        hp_r1_loss = F.margin_ranking_loss(hp_r_scores, hp_1_scores, target_one, margin=self.cfg.margin)
        hp_loss = hp_10_loss + hp_r1_loss

        icthp_10_loss = F.margin_ranking_loss(icthp_1_scores, icthp_0_scores, target_one, margin=self.cfg.margin, reduction="none")
        icthp_r1_loss = F.margin_ranking_loss(icthp_r_scores, icthp_1_scores, target_one, margin=self.cfg.margin, reduction="none")
        icthp_loss = icthp_10_loss.sum() + icthp_r1_loss.sum()

        if self.cfg.loss_ict:
            loss = self.cfg.ict_weight * ict_loss
        else:
            if self.cfg.loss_hp and self.cfg.loss_icthp:
                loss = self.cfg.hp_weight * hp_loss + self.cfg.icthp_weight * icthp_loss
            elif self.cfg.loss_hp:
                loss = self.cfg.hp_weight * hp_loss
            elif self.cfg.loss_icthp:
                loss = self.cfg.icthp_weight * icthp_loss

        return loss, hp_loss, icthp_loss

    def forward(self, model, batch, globel_step):
        image_0_features, image_1_features, image_r_features, easy_text_features, refine_text_features, hp_0_scores, hp_1_scores, hp_r_scores = self.get_features(
            model,
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name],
            batch[self.cfg.pixels_r_column_name],
            batch[self.cfg.easy_input_ids_column_name],
            batch[self.cfg.refine_input_ids_column_name],
            is_hp=True,
        )
        
        loss, hp_loss, icthp_loss = self.calc_loss(
            easy_text_features,
            refine_text_features,
            image_0_features,
            image_1_features,
            image_r_features,
            hp_0_scores,
            hp_1_scores,
            hp_r_scores,
            globel_step,
            batch[self.cfg.label_E1_column_name],
            batch[self.cfg.label_E2_column_name],
            batch[self.cfg.label_R1_column_name],
            batch[self.cfg.label_R2_column_name],
        )
        return loss, hp_loss, icthp_loss

