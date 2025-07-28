import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from hpsv2.src.open_clip import get_tokenizer
import torch.nn as nn

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
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)

    def forward(self, x):
        return self.layers(x)


def calculate_ict_hp_scores(image_score, prompt, tokenizer, reward_models, device):
    """Calculate ICT and HP scores for given image and prompt"""
    with torch.no_grad():
        ict_model, hp_backbone, hp_scorer = reward_models
        
        # ICT score calculation
        image_ict_features = ict_model.get_image_features(pixel_values=image_score)
        image_ict_features = image_ict_features / image_ict_features.norm(dim=-1, keepdim=True)
        
        # Process text input
        text_input_ids = tokenizer(prompt).to(device)
        text_features_ict = ict_model.get_text_features(text_input_ids)
        text_features_ict = text_features_ict / text_features_ict.norm(dim=-1, keepdim=True)
        
        ict_scores = text_features_ict @ image_ict_features.T
       
        # HP score calculation
        image_hp_backbone_features = hp_backbone.get_image_features(pixel_values=image_score)
        hp_scores = hp_scorer(image_hp_backbone_features)
        
        # Apply sigmoid to ensure HP scores are positive (0-1 range)
        hp_scores = torch.sigmoid(hp_scores)
        
        scores = {
            "ict_score": ict_scores.cpu().item() if ict_scores.dim() == 0 else ict_scores.cpu().squeeze().item(),
            "hp_score": hp_scores.cpu().item() if hp_scores.dim() == 0 else hp_scores.cpu().squeeze().item(),
        }
        
    return scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Image groups with corresponding prompts
    image_groups = [
        {
            "prompt": "a tiny dragon taking a bath in a teacup",
            "images": ["tiny_dragon_teacup_I1.jpg", "tiny_dragon_teacup_I2.jpg", "tiny_dragon_teacup_I3.png"],
            "group_name": "Tiny Dragon Teacup"
        },
        {
            "prompt": "a Ferrari car that is made out of wood", 
            "images": ["Ferrari_car_I1.jpg", "Ferrari_car_I2.jpg", "Ferrari_car_I3.png"],
            "group_name": "Wood Ferrari Car"
        }
    ]
    
    print(f"Processing {len(image_groups)} image groups")
    
    # Load CLIP models
    pretrained_model_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    
    # Load models and prepare preprocessor
    clip_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path)
    preprocess_val = lambda img: clip_processor(images=img, return_tensors="pt")["pixel_values"]
    
    # Load ICT model
    print("Loading ICT model...")
    ict_model = CLIPModel.from_pretrained(pretrained_model_name_or_path)
    ictmodel_path ="./ICTHP_models/ICT"
    checkpoint_path = f"{ictmodel_path}/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    ict_model.load_state_dict(state_dict, strict=False)
    ict_model = ict_model.to(device)
    ict_model.eval()
    
    # Load CLIP and HP models
    print("Loading HP model...")
    hp_backbone = CLIPModel.from_pretrained(pretrained_model_name_or_path)
    hp_scorer = MLP(hidden_dim=1024)
    
    hpmodel_path ="./ICTHP_models/HP"
    hp_backbone_checkpoint_path = f"{hpmodel_path}/hp_backbone/pytorch_model.bin"
    hp_scorer_checkpoint_path = f"{hpmodel_path}/hp_scorer/mlp_pytorch_model.bin"
    
    hp_backbonestate_dict = torch.load(hp_backbone_checkpoint_path, map_location="cpu")
    hp_scorer_state_dict = torch.load(hp_scorer_checkpoint_path, map_location="cpu")
    
    hp_backbone.load_state_dict(hp_backbonestate_dict, strict=False)
    hp_scorer.load_state_dict(hp_scorer_state_dict, strict=False)
    
    hp_backbone = hp_backbone.to(device)
    hp_scorer = hp_scorer.to(device)
    hp_backbone.eval()
    hp_scorer.eval()
    
    # Get tokenizer
    tokenizer = get_tokenizer('ViT-H-14')
    
    reward_models = (ict_model, hp_backbone, hp_scorer)
    
    print("\nStarting score calculation...")
    print("="*80)
    
    # Process each image group
    for group_idx, group in enumerate(image_groups):
        print(f"\n[Group {group_idx + 1}] {group['group_name']}")
        print(f"Prompt: {group['prompt']}")
        print("="*60)
        
        group_results = []
        
        # Process each image in current group
        for image_file in group['images']:
            try:
                # Load image
                image_path = os.path.join(".", image_file)
                if not os.path.exists(image_path):
                    print(f"    Warning: Image file not found - {image_path}")
                    continue
                    
                image = Image.open(image_path).convert("RGB")
                image_score = preprocess_val(image).to(device)
                
                # Calculate scores
                scores = calculate_ict_hp_scores(image_score, group['prompt'], tokenizer, reward_models, device)
                
                # Save results
                result = {
                    'image': image_file,
                    'ict': scores['ict_score'],
                    'hp': scores['hp_score']
                }
                group_results.append(result)
                
                # Print individual image results
                print(f"      {image_file}")
                print(f"      ICT Score: {scores['ict_score']:.4f}")
                print(f"      HP Score:  {scores['hp_score']:.4f}")
                print()
                
            except Exception as e:
                print(f"  Error processing {image_file}: {e}")

        print("-" * 60)
    


if __name__ == '__main__':
    main()
