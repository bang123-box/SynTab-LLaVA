# Adopted from https://github.com/alibaba/conv-llava/blob/main/llava/model/multimodal_encoder/clip_encoder.py
import torch
import torch.nn as nn
import argparse
from transformers import CLIPImageProcessor
from transformers import ConvNextModel, ConvNextConfig
from transformers.models.convnext.modeling_convnext import ConvNextStage


class ConvNeXtVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        
        if not delay_load:
            self.load_model()
        else:
            print(f"deloy_load vision tower is: {self.vision_tower_name}")
            self.cfg_only = ConvNextConfig.from_pretrained(
                self.vision_tower_name)

    def load_model(self, device_map=None):
        print(f"entering load model, load {self.vision_tower_name}")
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name)
        self.vision_tower = ConvNextModel.from_pretrained(
            self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # Get the embeddings of the image
                embedding_output = self.vision_tower.embeddings(image.unsqueeze(0))

                # Get the image features
                image_feature = self.vision_tower.encoder(embedding_output,
                                                        output_hidden_states=True,
                                                        return_dict=True)
                image_feature = image_feature.hidden_states[-1].permute(0, 2, 3, 1)
                image_feature = image_feature.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)

                image_features.append(image_feature)
        else:
            embedding_output = self.vision_tower.embeddings(images)
            image_features = self.vision_tower.encoder(embedding_output,
                                                       output_hidden_states=True,
                                                       return_dict=True)
            image_features = image_features.hidden_states[-1].permute(0, 2, 3, 1)
            image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)

        return image_features

    def make_layers_trainable_after_stage(self, stage_index, layer_index=0):
        for i, stage in enumerate(self.vision_tower.encoder.stages):
            if i == stage_index:
                if layer_index == 0:
                    stage.downsampling_layer.requires_grad_(True)
                for idx, layer in enumerate(stage.layers):
                    if idx >= layer_index:
                        for param in layer.parameters():
                            param.requires_grad = True
            if i > stage_index:
                stage.downsampling_layer.requires_grad_(True)
                for layer in stage.layers:
                    for param in layer.parameters():
                        param.requires_grad = True

    def set_crop_size(self, new_size):
        size_dict = {'height': new_size, 'width': new_size}
        self.image_processor.crop_size = size_dict
        self.image_processor.size = {"shortest_edge": new_size}
        self.vision_tower.config.image_size = new_size

    def add_stage(self, depths=3, hidden_dims=3072):
        self.vision_tower.encoder.stages.append(ConvNextStage(self.config, self.hidden_size, hidden_dims, depth=depths))
        self.vision_tower.config.depths.append(depths)
        self.vision_tower.config.hidden_sizes.append(hidden_dims)
        self.vision_tower.config.stage_names.append('stage5')
        self.vision_tower.config.out_features = ['stage5']
        self.vision_tower.config.out_indices = [5]
        self.vision_tower.config.num_stages += 1
        self.vision_tower.config._name_or_path = ''

    def save_config(self, path):
        self.vision_tower.config.save_pretrained(path)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_sizes[-1]

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // 64) ** 2

    @property
    def crop_size(self):
        return self.image_processor.crop_size