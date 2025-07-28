from dataclasses import dataclass
from transformers import CLIPModel
import torch.nn as nn
import torch
from trainer.models.base_model import BaseModelConfig
import os

@dataclass
class ICTHPModelConfig(BaseModelConfig):
    _target_: str = "trainer.models.icthp_model.ICTHPModel"
    pretrained_model_name_or_path: str = "openai/clip-vit-base-patch32"
    hidden_dim: int = 1024
    fix_rate: float = 0.8
    ict_model_path: str = ""
    train_hp: bool = False

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def set_grad_with_rate(model, fix_rate):
    if not 0 <= fix_rate <= 1:
        raise ValueError("fix_rate should be between 0 and 1")
    
    text_layers = model.text_model.encoder.layers + [model.text_model.final_layer_norm]
    for layer in text_layers:
        for param in layer.parameters():
            param.requires_grad = False
    
    image_layers = model.vision_model.encoder.layers + [model.vision_model.post_layernorm]
    image_fix_num = int(len(image_layers) * fix_rate)
    for i, layer in enumerate(image_layers):
        for param in layer.parameters():
            param.requires_grad = i >= image_fix_num
    
    for param in model.text_projection.parameters():
        param.requires_grad = False
    for param in model.visual_projection.parameters():
        param.requires_grad = False

class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, x):
        return self.layers(x)

    def save_pretrained(self, save_directory):
        torch.save(self.state_dict(), os.path.join(save_directory, 'mlp_pytorch_model.bin'))

    def from_pretrained(self, load_directory):
        state_dict = torch.load(os.path.join(load_directory, 'mlp_pytorch_model.bin'))
        self.load_state_dict(state_dict)

class ICTHPModel(nn.Module):
    def __init__(self, cfg: ICTHPModelConfig):
        super().__init__()
        self.ict_model = CLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)
        
        if cfg.train_hp:
            checkpoint_path = f"{cfg.ict_model_path}/pytorch_model.bin"
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.ict_model.load_state_dict(state_dict, strict=False)
        
        self.hp_backbone = CLIPModel.from_pretrained(cfg.pretrained_model_name_or_path)
        self.hp_scorer = MLP(hidden_dim=cfg.hidden_dim)
        self.fix_rate = cfg.fix_rate
        self.train_hp = cfg.train_hp

        if cfg.train_hp:
            set_requires_grad(self.ict_model, False)
            if cfg.fix_rate == 0:
                set_requires_grad(self.hp_backbone, False)
            else:
                set_grad_with_rate(self.hp_backbone, cfg.fix_rate)
            set_requires_grad(self.hp_scorer, True)
        else:
            if cfg.fix_rate == 0:
                set_requires_grad(self.hp_backbone, False)
            else:
                set_grad_with_rate(self.hp_backbone, cfg.fix_rate)
            set_requires_grad(self.hp_scorer, False)

    def get_text_features(self, text_inputs, model="ict"):
        if model == "ict":
            return self.ict_model.get_text_features(text_inputs)
        elif model == "hp":
            return self.hp_backbone.get_text_features(text_inputs)
        else:
            raise ValueError("Invalid model type. Choose 'ict' or 'hp'.")

    def get_image_features(self, image_inputs, model="ict"):
        if model == "ict":
            return self.ict_model.get_image_features(image_inputs)
        elif model == "hp":
            return self.hp_backbone.get_image_features(image_inputs)
        else:
            raise ValueError("Invalid model type. Choose 'ict' or 'hp'.")

    def forward(self, text_inputs=None, image_inputs=None, is_hp=None):
        outputs = ()
        if text_inputs is not None:
            outputs += self.get_text_features(text_inputs, model="ict"),
        if image_inputs is not None:
            outputs += self.ict_model.get_image_features(image_inputs),
        if is_hp is not None:
            hp_backbone_image_features = self.get_image_features(image_inputs, model="hp")
            outputs += self.hp_scorer(hp_backbone_image_features),
        return outputs

    @property
    def logit_scale(self):
        return self.hp_backbone.logit_scale

    def save(self, path):
        if self.train_hp:
            hp_path = os.path.join(path, "hp_scorer")
            os.makedirs(hp_path, exist_ok=True)
            self.hp_scorer.save_pretrained(hp_path)
            if self.fix_rate != 0:
                hp_backbone_path = os.path.join(path, "hp_backbone")
                os.makedirs(hp_backbone_path, exist_ok=True)
                self.hp_backbone.save_pretrained(hp_backbone_path)
        else:
            ict_path = os.path.join(path, "ict_model")
            os.makedirs(ict_path, exist_ok=True)
            self.ict_model.save_pretrained(ict_path)

        

