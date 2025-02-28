from transformers import Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration
import torch
import torch.nn as nn
import sys
import pdb 

'''
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/MiniGPT-4")
from minigpt4.common.utils import now
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
'''

from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig, Blip2Processor, Blip2ForConditionalGeneration
# sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
sys.path.append("models/LLaVA_modified/LLaVA")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.mm_utils import KeywordsStoppingCriteria

try:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
except:
    pass


class Blip2ModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.model.language_model.lm_head.weight.requires_grad = True

    def generate(self, input_ids, attention_mask, pixel_values, labels=None, min_new_tokens=1, max_new_tokens=100):
        
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)

        return generated_ids

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs
    

class InstructBlipModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", max_position_embeddings=2048)


    def generate(self, qformer_input_ids, input_ids, attention_mask, pixel_values, labels=None, min_new_tokens=1, max_new_tokens=100):
        
        generated_ids = self.model.generate(qformer_input_ids=qformer_input_ids,input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, min_new_tokens=min_new_tokens, max_new_tokens=max_new_tokens)

        return generated_ids

    def forward(self, qformer_input_ids, input_ids, attention_mask, pixel_values, labels=None):
        
        outputs = self.model(qformer_input_ids=qformer_input_ids, input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs


class DinoProjector(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model = DINO.load_pretrained("dino_vit_base8_384", num_classes=0, use_projection=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, pixel_values):
        with torch.no_grad():
            features = self.model(pixel_values)
        return features



class LlavaModelLiteInterface(nn.Module):

    def __init__(self, args):
        super().__init__()
        model_path = "liuhaotian/llava-v1.5-7b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            encode_lite=True,
            # load_8bit=True
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            use_lite=True,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria], 
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None,):
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask, 
            use_lite=True)
        # pdb.set_trace()

        return outputs
    
class LlavaModelDinoInterface(nn.Module):

    def __init__(self, args):
        super().__init__()
        # model_path = "liuhaotian/llava-v1.5-7b"
        model_path = "liuhaotian/llava-v1.5-13b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            use_dino=True,
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None,):
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()

        return outputs


class CLIPEncoderCameraPolygon(nn.Module):
    def __init__(self, vision_tower):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
    
    def load_model(self, device_map=None):
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

        encoder = nn.TransformerEncoderLayer(
            1024, 4, batch_first=True
        )
        self.transformer_encoder_block = nn.TransformerEncoder(
            encoder, 2
        )

        # self.feat_projection = nn.Sequential(nn.Linear(768, 1024), nn.Tanh())

        self.cls_cam = nn.Parameter(torch.randn(1, 1, 1024))
        self.cls_poly = nn.Parameter(torch.randn(1, 1, 1024))
        self.cls_im = nn.Parameter(torch.randn(1, 1, 1024))

        self.cam_projection = nn.Sequential(nn.Linear(3, 1024), nn.Tanh())
        self.poly_projection = nn.Sequential(nn.Linear(2, 1024), nn.Tanh())

        self.is_loaded = True
        self.loss = nn.MSELoss()


    def forward(self, image_processed, cameraposition, polygon, gt_im_features=None):
        # with torch.no_grad():
        outputs = self.vision_tower(image_processed.to(device=self.device, dtype=self.dtype))

        # pdb.set_trace()
        image_features = outputs[0]

        # image_features = self.transformer_encoder_block(image_features)
        image_features = image_features[:, 1:, :]

        cam_feats = self.cam_projection(cameraposition.to(dtype=self.dtype, device=self.device))
        poly_feats = self.poly_projection(polygon.to(dtype=self.dtype, device=self.device)).squeeze()
        
        cat_feats = torch.cat([self.cls_cam, cam_feats, self.cls_poly, poly_feats, self.cls_im, image_features.unsqueeze(0)], dim=1)

        image_features = self.transformer_encoder_block(cat_feats)

        image_features = image_features[:, 0:1, :]

        if gt_im_features is not None:
            loss = self.loss(image_features, gt_im_features)
        
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


class LlavaModelCLIPCameraPolygonInterface(nn.Module):

    def __init__(self, args):
        super().__init__()
        # model_path = "liuhaotian/llava-v1.5-7b"
        model_path = "liuhaotian/llava-v1.5-13b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            use_clip_camera_polygon=True,
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None, camera_pos=None, polygon=None, gt_im_features=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            camera_pos=camera_pos,
            polygon=polygon,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None, camera_pos=None, polygon=None, gt_im_features=None):
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            gt_im_features=gt_im_features,
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask,
            camera_pos=camera_pos,
            polygon=polygon, 
            output_im_features=True)
        # pdb.set_trace()

        return outputs



    
class LlavaModelDinoCameraPolygonInterface(nn.Module):

    def __init__(self, args):
        super().__init__()
        # model_path = "liuhaotian/llava-v1.5-7b"
        model_path = "liuhaotian/llava-v1.5-13b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            use_dino_camera_polygon=True,
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None, camera_pos=None, polygon=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            camera_pos=camera_pos,
            polygon=polygon,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None, camera_pos=None, polygon=None):
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask,
            camera_pos=camera_pos,
            polygon=polygon)
        # pdb.set_trace()

        return outputs
    
class LlavaModelDinoInterface_FeatureLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        # model_path = "liuhaotian/llava-v1.5-7b"
        model_path = "liuhaotian/llava-v1.5-13b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            use_dino_camera_polygon=True,
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None, camera_pos=None, polygon=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            camera_pos=camera_pos,
            polygon=polygon,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None, output_im_features=True, gt_im_features=None, camera_pos=None, polygon=None):
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask, 
            camera_pos=camera_pos,
            polygon=polygon,
            output_im_features=output_im_features)
        # pdb.set_trace()

        return outputs


class LlavaModel_13B_Interface(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        model_path = "liuhaotian/llava-v1.5-13b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            # load_8bit=True
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values=None, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)
        
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        # pdb.set_trace()

        return output_ids

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None,):
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)

        outputs =  self.model(
            input_ids,
            images=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()
        
        return outputs
    
class Robopoint_13B_Interface(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        model_path = "wentao-yuan/robopoint-v1-vicuna-v1.5-13b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            # load_8bit=True
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values=None, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)
        
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        # pdb.set_trace()

        return output_ids

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None,):
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)

        outputs =  self.model(
            input_ids,
            images=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()
        
        return outputs

class LlavaNext_13B_Interface(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.image_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values=None, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)
        
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        # pdb.set_trace()

        return output_ids

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None,):
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)

        outputs =  self.model(
            input_ids,
            images=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()
        
        return outputs




class LLaVA_OV_Interface(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto")
        self.image_processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

        self.tokenizer = self.image_processor.tokenizer
        self.max_new_tokens = args['max_new_tokens']

    
    def generate(self, input_ids, pixel_values=None, attention_mask=None, labels=None, image_sizes=None):
        
        #pdb.set_trace()
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=self.max_new_tokens,
            attention_mask=attention_mask,
            image_sizes=image_sizes
        )
        # pdb.set_trace()

        return output_ids

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None, image_sizes=None):
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)

        outputs =  self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask,
            image_sizes=image_sizes)
        # pdb.set_trace()
        
        return outputs


class LlavaModelInterface(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        model_path = "liuhaotian/llava-v1.5-7b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            # load_8bit=True
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None,):
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()
        
        return outputs
    

class LlavaModel_16V_Interface(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        model_path = "liuhaotian/llava-v1.6-vicuna-13b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            # load_8bit=True
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["\n###", "###"]

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        #pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None,):
        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()
        
        return outputs


class GPT4_Interface(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = ""

    
    def get_caption(image_path, prompt, api_key=""):
        # Getting the base64 string
        base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                ]
            }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        return response

    def forward(self, image_path, prompt):
        outputs = self.get_caption(image_path, prompt)
        return outputs

    def generate(self, input_ids):
        outputs = self.get_caption(image_path, prompt)
        return outputs


class Llava3DModelInterface(nn.Module):

    def __init__(self, args):

        super().__init__()
        model_path = "liuhaotian/llava-v1.5-7b"

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            encoder_multiview=True,
            # load_8bit=True
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

        self.keywords = ["###", " \n###"]

    def generate(self, input_ids, pixel_values, camera_pos, attention_mask=None, labels=None):
        stopping_criteria = KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)
        # pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            camera_pos = camera_pos,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        return output_ids

    def forward(self, input_ids, pixel_values, camera_pos, attention_mask, labels=None,):
        outputs =  self.model.forward(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            camera_pos = camera_pos,
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()
        # pdb.set_trace()
        return outputs


class LlavaModelAggImgInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_path = "liuhaotian/llava-v1.5-7b"

        _, self.model, _, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path)
        )

        self.temperature = args['temperature']
        self.top_p = args['top_p']
        self.num_beams = args['num_beams']
        self.max_new_tokens = args['max_new_tokens']

    def generate(self, input_ids, pixel_values, attention_mask=None, labels=None):

        # pdb.set_trace()
        output_ids = self.model.generate(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            top_p=self.top_p,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
        )

        return output_ids

    def forward(self, input_ids, pixel_values, attention_mask, labels=None,):

        outputs =  self.model(
            input_ids,
            images=pixel_values.to(self.model.dtype),
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)

        return outputs


class Blip2ModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        #config = InstructionBlipConfig.from_pretrained("Salesforce/instructblip-vicuna-7b")
        #config.max_position_embeddings = 2048
        #self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch.float16)
        
        
        self.model = self.model.half()

        self.image_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.tokenizer = None

        self.max_new_tokens = args['max_new_tokens']

    def generate(self, input_ids, attention_mask, pixel_values, labels=None):
        
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        return generated_ids

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs
    
class InstructBlipModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        
        self.model = self.model.half()

        self.image_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        self.tokenizer = None

    def generate(self, input_ids, qformer_input_ids, attention_mask, qformer_attention_mask, pixel_values, labels=None):
        
        generated_ids = self.model.generate(input_ids=input_ids, qformer_input_ids=qformer_input_ids, attention_mask=attention_mask, qformer_attention_mask=qformer_attention_mask, pixel_values=pixel_values,
                                                do_sample=False, num_beams=5, max_length=256, min_length=1, top_p=0.9, repetition_penalty=1.5, length_penalty=1.0, temperature=1,)

        return generated_ids

    def forward(self, input_ids, qformer_input_ids, attention_mask, qformer_attention_mask, pixel_values, labels=None):
        
        outputs = self.model(input_ids=input_ids, qformer_input_ids=qformer_input_ids, attention_mask=attention_mask, qformer_attention_mask=qformer_attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs

class KOSMOS2ModelInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224")
        self.image_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        
    def generate(self, input_ids, attention_mask, pixel_values, labels=None):
        pass
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        pass

class CodeLLamaInterface(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-13b-hf")
        self.max_new_tokens = args['max_new_tokens']

        self.tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-13b-hf")
        self.image_processor = None

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate(self, input_ids):
        generated_ids = self.model.generate(input_ids=input_ids, max_new_tokens=self.max_new_tokens)
        return generated_ids


class Molmo7BInterface(nn.Module):

    def __init__(self, args):
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        pass

    def forward(self, input_ids, attention_mask, labels):
        pass

    def generate(self, input_ids):
        pass




""" MiniGPT4 interface, in progress
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def get_multimodal_prompt(model, img, prompt):
    prompt_segs = prompt.split('<ImageHere>')
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).input_ids.cuda()
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]

    seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [seg_embs[0].repeat(len(img), 1, 1), img, seg_embs[1].repeat(len(img), 1, 1)]
    #mixed_embs = [emb for pair in zip(seg_embs[:-1], [img]) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

class MiniGPT4Interface(nn.Module):
    def __init__(self, args):
        super().__init__()
        cfg = Config(args)
        cfg.pretty_print()
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        model = model.cuda().eval()

        stop_words_ids = [torch.tensor([835]).cuda(),
                        torch.tensor([2277, 29937]).cuda()]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    
    def forward(self, pixel_values, prompts, labels=None):
        curr_img_feature, _ = model.encode_img(curr_processed)
        curr_multimodal_embd = get_multimodal_prompt(model, curr_img_feature, prompt)
        outputs = self.model(input_embeds=input_embeds, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)

        return outputs

    def generate(self, input_embeds, 
                 max_new_tokens=100, 
                 stopping_criteria=None, 
                 num_beams=None, do_sample=False,
                 min_length=1, 
                 top_p=1, 
                 repetition_penalty=None, 
                 length_penalty=None, 
                 temperature=0.2):
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        return outputs
"""