from transformers import Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration
import torch
import torch.nn as nn
import sys
import pdb 
from huggingface_hub import PyTorchModelHubMixin

from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

sys.path.append("models/LLaVA_modified/LLaVA")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.mm_utils import KeywordsStoppingCriteria

try:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
except:
    pass



class LlavaModel_13B_Interface(nn.Module, PyTorchModelHubMixin):

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
        self.output_hidden_states = args.get('output_hidden_states', False)

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
            stopping_criteria=[stopping_criteria],
            output_hidden_states=self.output_hidden_states
        )

        # pdb.set_trace()

        return output_ids

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None,):
        if pixel_values is not None:
            pixel_values.to(self.model.dtype)

        outputs =  self.model(
            input_ids,
            images=pixel_values,
            output_hidden_states=self.output_hidden_states,
            return_dict=True,
            labels=labels,
            attention_mask=attention_mask)
        # pdb.set_trace()
        
        return outputs



