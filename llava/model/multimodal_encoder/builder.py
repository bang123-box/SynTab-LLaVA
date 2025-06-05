import os
import torch 
import torch.nn as nn
from .convnext_encoder import ConvNeXtVisionTower
from .clip_encoder import CLIPVisionTower

class Mix_VisionTower(nn.Module):
    def __init__(self, highres_vision_tower, lowres_vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.highres_vision_tower_name, self.lowres_vision_tower_name = highres_vision_tower, lowres_vision_tower
        print(highres_vision_tower, lowres_vision_tower)
        self.highres_vision_tower = None if not highres_vision_tower else ConvNeXtVisionTower(highres_vision_tower, args, delay_load)
        self.lowres_vision_tower = None if not lowres_vision_tower else CLIPVisionTower(lowres_vision_tower, args, delay_load)
        
        if self.highres_vision_tower == None and self.lowres_vision_tower == None:
            raise ("highres_vision_tower and lowres_vision_tower can not be None mean time.")
        if not delay_load:
            self.is_loaded = True

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{}  and {} is already loaded, `load_model` called again, skipping.'.format(self.highres_vision_tower_name, self.lowres_vision_tower_name))
            return
        if self.highres_vision_tower:
            self.highres_vision_tower.load_model(device_map)
        if self.lowres_vision_tower:
            self.lowres_vision_tower.load_model(device_map)
        self.is_loaded = True
    
    @torch.no_grad()
    def forward(self, highres_images, lowres_images):
        features = []
        if self.highres_vision_tower:
            features.append(self.highres_vision_tower(highres_images))
        if self.lowres_vision_tower:
            features.append(self.lowres_vision_tower(lowres_images))
        return torch.cat(features, dim=-1).to(dtype=features[0].dtype, device=features[0].device)
    
    @property
    def hidden_size(self):
        # size = 0
        # if self.highres_vision_tower:
        #     size += self.highres_vision_tower.hidden_size
        # if self.lowres_vision_tower:
        #     size += self.lowres_vision_tower.hidden_size
        return (self.highres_vision_tower.hidden_size if self.highres_vision_tower else 0 ) +\
                (self.lowres_vision_tower.hidden_size if self.lowres_vision_tower else 0 )    
    
    @property
    def image_processor(self):
        preprocess = []
        if self.highres_vision_tower:
            preprocess.append(self.highres_vision_tower.image_processor)
        if self.lowres_vision_tower:
            preprocess.append(self.lowres_vision_tower.image_processor)
        if len(preprocess) == 1:
            preprocess.append(preprocess[0])
        return preprocess        
    

def build_vision_tower(vision_tower_cfg, **kwargs):
    highres_vision_tower = getattr(vision_tower_cfg, 'mm_highres_vision_tower', getattr(vision_tower_cfg, 'highres_vision_tower', None))
    lowres_vision_tower = getattr(vision_tower_cfg, 'mm_lowres_vision_tower', getattr(vision_tower_cfg, 'lowres_vision_tower', None))
    return Mix_VisionTower(highres_vision_tower, lowres_vision_tower, args=vision_tower_cfg, **kwargs)
    # is_absolute_path_exists = os.path.exists(vision_tower)
    # use_s2 = getattr(vision_tower_cfg, 's2', False)
    # if is_absolute_path_exists and "convnext" in vision_tower.lower():
    #     return ConvNeXtVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
    #     if use_s2:
    #         return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
    #     else:
    #         return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # raise ValueError(f'Unknown vision tower: {vision_tower}')
