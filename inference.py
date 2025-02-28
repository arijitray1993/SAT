import models.model_interface as models

from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
from huggingface_hub import hf_hub_download


if __name__=="__main__":

    config = {
        'args':{
            'temperature': 0,
            'top_p': None,
            'num_beams': 1,
            'max_new_tokens': 768
        },
    }
    load_model = models.LlavaModel_13B_Interface(**config)
    lora_model = PeftModel.from_pretrained(load_model, "array/sat-dynamic-13b")


