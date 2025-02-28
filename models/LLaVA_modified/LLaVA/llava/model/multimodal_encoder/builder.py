import os
from .clip_encoder import CLIPVisionTower
from .clip_encoder import DinoEncoder, DinoEncoderCameraPolygon, CLIPEncoderCameraPolygon
        

def build_vision_tower(vision_tower_cfg, **kwargs):

    if kwargs.get("use_dino", False):
        return DinoEncoder()
    elif kwargs.get("use_dino_camera_polygon", False):
        return DinoEncoderCameraPolygon()
    elif kwargs.get("use_clip_camera_polygon", False):
        vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
        is_absolute_path_exists = os.path.exists(vision_tower)
        return CLIPEncoderCameraPolygon(vision_tower=vision_tower)
    else:
        vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
        is_absolute_path_exists = os.path.exists(vision_tower)
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
