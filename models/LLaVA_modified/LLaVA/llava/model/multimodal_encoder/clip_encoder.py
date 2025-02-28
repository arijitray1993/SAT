import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTextModel
from transformers import AutoImageProcessor, AutoModel
import pdb

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

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
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class DinoEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.is_loaded = False
    
    def load_model(self, device_map=None):
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.vision_tower = AutoModel.from_pretrained("facebook/dinov2-base",)

        self.vision_tower.requires_grad_(False)

        encoder = nn.TransformerEncoderLayer(
            768, 4, batch_first=True
        )
        self.transformer_encoder_block = nn.TransformerEncoder(
            encoder, 2
        )

        self.feat_projection = nn.Sequential(nn.Linear(768, 1024), nn.Tanh())

        self.is_loaded = True

    def forward(self, image_processed):
        with torch.no_grad():
            outputs = self.vision_tower(image_processed.to(device=self.device, dtype=self.dtype))

        image_features = outputs[0]
        # pdb.set_trace()

        image_features = self.transformer_encoder_block(image_features)

        image_features = image_features[:, 0, :]

        image_features = self.feat_projection(image_features)

        return image_features.unsqueeze(1)
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return 1
    

class DinoEncoderCameraPolygon(nn.Module):
    def __init__(self,):
        super().__init__()
        self.is_loaded = False
    
    def load_model(self, device_map=None):
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.vision_tower = AutoModel.from_pretrained("facebook/dinov2-base",)

        self.vision_tower.requires_grad_(False)

        encoder = nn.TransformerEncoderLayer(
            768, 4, batch_first=True
        )
        self.transformer_encoder_block = nn.TransformerEncoder(
            encoder, 2
        )

        self.feat_projection = nn.Sequential(nn.Linear(768, 1024), nn.Tanh())

        self.cls_cam = nn.Parameter(torch.randn(1, 768))
        self.cls_poly = nn.Parameter(torch.randn(1, 768))
        self.cls_im = nn.Parameter(torch.randn(1, 768))

        self.cam_projection = nn.Sequential(nn.Linear(3, 768), nn.Tanh())
        self.poly_projection = nn.Sequential(nn.Linear(2, 768), nn.Tanh())

        self.is_loaded = True

        

    def forward(self, image_processed, cameraposition, polygon):
        with torch.no_grad():
            outputs = self.vision_tower(image_processed.to(device=self.device, dtype=self.dtype))

        image_features = outputs[0]

        image_features = self.transformer_encoder_block(image_features)
        image_features = image_features[:, 0, :]

        cam_feats = self.cam_projection(cameraposition)
        poly_feats = self.poly_projection(polygon)
        pdb.set_trace()
        cat_feats = torch.cat([self.cls_cam.repeat(image_features.size(0), 1), image_features, self.cls_poly.repeat(image_features.size(0), 1), poly_feats, self.cls_im.repeat(image_features.size(0), 1), cam_feats], dim=0)

        image_features = self.feat_projection(image_features)

        return image_features.unsqueeze(1)
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return 1


class CLIPEncoderCameraPolygon(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
    
    def load_model(self, device_map=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        encoder = nn.TransformerEncoderLayer(
            1024, 4, batch_first=True
        )
        self.transformer_encoder_block = nn.TransformerEncoder(
            encoder, 2
        )

        # self.feat_projection = nn.Sequential(nn.Linear(768, 1024), nn.Tanh())

        self.cls_cam = nn.Parameter(torch.randn(1, 1024))
        self.cls_poly = nn.Parameter(torch.randn(1, 1024))
        self.cls_im = nn.Parameter(torch.randn(1, 1024))

        self.cam_projection = nn.Sequential(nn.Linear(3, 1024), nn.Tanh())
        self.poly_projection = nn.Sequential(nn.Linear(2, 1024), nn.Tanh())

        self.is_loaded = True

        

    def forward(self, image_processed, cameraposition, polygon):
        # with torch.no_grad():
        outputs = self.vision_tower(image_processed.to(device=self.device, dtype=self.dtype))

        # pdb.set_trace()
        image_features = outputs[0]

        # image_features = self.transformer_encoder_block(image_features)
        image_features = image_features[:, 0, :]

        cam_feats = self.cam_projection(cameraposition.to(dtype=self.dtype, device=self.device))
        poly_feats = self.poly_projection(polygon.to(dtype=self.dtype, device=self.device)).squeeze()
        
        cat_feats = torch.cat([self.cls_cam, cam_feats, self.cls_poly, poly_feats, self.cls_im, image_features], dim=0)

        cat_feats = cat_feats.unsqueeze(0)

        image_features = self.transformer_encoder_block(cat_feats)

        image_features = image_features[:, 0:1, :]
        # pdb.set_trace()
        return image_features
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return 1
    

class CLIPEncoderCameraPolygonText(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
    
    def load_model(self, device_map=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        encoder = nn.TransformerEncoderLayer(
            1024, 4, batch_first=True
        )
        self.transformer_encoder_block = nn.TransformerEncoder(
            encoder, 2
        )

        # self.feat_projection = nn.Sequential(nn.Linear(768, 1024), nn.Tanh())

        self.cls_cam = nn.Parameter(torch.randn(1, 1024))
        self.cls_poly = nn.Parameter(torch.randn(1, 1024))
        self.cls_im = nn.Parameter(torch.randn(1, 1024))

        self.cam_projection = nn.Sequential(nn.Linear(3, 1024), nn.Tanh())
        self.poly_projection = nn.Sequential(nn.Linear(2, 1024), nn.Tanh())

        self.is_loaded = True

        

    def forward(self, image_processed, cameraposition, polygon):
        # with torch.no_grad():
        outputs = self.vision_tower(image_processed.to(device=self.device, dtype=self.dtype))

        # pdb.set_trace()
        image_features = outputs[0]

        # image_features = self.transformer_encoder_block(image_features)
        image_features = image_features[:, 0, :]

        cam_feats = self.cam_projection(cameraposition.to(dtype=self.dtype, device=self.device))
        poly_feats = self.poly_projection(polygon.to(dtype=self.dtype, device=self.device)).squeeze()
        
        cat_feats = torch.cat([self.cls_cam, cam_feats, self.cls_poly, poly_feats, self.cls_im, image_features, ], dim=0)

        cat_feats = cat_feats.unsqueeze(0)

        image_features = self.transformer_encoder_block(cat_feats)

        image_features = image_features[:, 0:1, :]
        # pdb.set_trace()
        return image_features
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return 1