# SAT: Spatial Aptitude Training for Multimodal Language Models
Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, Kate Saenko

**Note: This code release is a work in progress - we will update it with full instructions soon.**


[Project Page](https://arijitray1993.github.io/SAT/)
[Paper](https://arxiv.org/abs/2412.07755)

![SAT Data](https://arijitray1993.github.io/SAT/SAT_webpage/static/images/sat_teaser.png)


## Pre-requisities

Clone this repo and create a conda environment:

```
git clone https://github.com/arijitray1993/SAT.git
conda env create -n sat_env python=3.10
conda activate sat_conda
pip install -r requirements.txt
mkdir checkpoints/
```

## Run Inference with our best SAT model

```python

# choose the one you want
CHECKPOINT_NAME = "array/sat-dynamic-13b" # trained on SAT dynnamic data (numbers used in ArXiV paper)

import models.model_interface as models
from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
from huggingface_hub import hf_hub_download

from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        get_model_name_from_path,
        KeywordsStoppingCriteria,
    )
from llava.mm_utils import expand2square
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

## Load the model
config = {
    'args':{
        'temperature': 0,
        'top_p': None,
        'num_beams': 1,
        'max_new_tokens': 768
    },
}
load_model = models.LlavaModel_13B_Interface(**config)
lora_model = PeftModel.from_pretrained(load_model, CHECKPOINT_NAME)
lora_model = lora_model.half()
model = lora_model.merge_and_unload()


## Process the data, assumes you have images_batch of PIL images and prompt which is a text string query like "Is the car to the left or right of pedestrian?"
images = []
for image in images_batch: 
    # list of images for a prompt or prompt batch. Even if one prompt in a batch requires 2 images, this list should be flattened.
    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
    # pdb.set_trace()
    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    images.append(image)

pixel_values = torch.stack(images, dim=0)

input_ids = []
attention_mask = []
input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
input_ids.append(input_id)
attention_mask.append(torch.ones_like(input_id))

input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    'pixel_values': pixel_values,
}

generated_ids = model(**inputs)
generated_ids[generated_ids==-200] = 1
generated_text = self.test_dataloader.dataset.batch_decode(generated_ids, skip_special_tokens=True)
```

## Run evals on CVBench and BLINK

Run:

`python -m accelerate.commands.launch main.py exp_name=llava_mixdata_IT_dynamicreasoning_MLMBench`


## Get the SAT Data
Follow instructions here: https://huggingface.co/datasets/array/SAT 

### (Needed for training) Download the LLaVA Instruct Tune data
Follow instructions here: https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning 


## Training

### Prepare accelerate to use deepspeed stage-2

`accelerate config`

Choose to use deepspeed, stage 2, gradient accumulation to 1, number GPUs to use based on your environment, everything else default. 

Next, run:

`python -m accelerate.commands.launch main.py exp_name=llava_mixdata_IT_dynamicreasoning`


The enrtire training config (located at `confs/exp_name/llava_mixdata_IT_dynamicreasoning.yaml`) should be self explanatory. 
You can extend the tuning by: 
- defining a new dataset in custom_datasets/dataloaders.py. 
- defining a new model in models/model_interface.py
- define the variables that the model should take as input from the output of your `collate_fn` in the dataloader in `model_input_choice:` in the `.yaml` file for your training. 


More instructions will be updated soon. 


## BibTeX

If you use this code/data, please cite:

```
@misc{ray2025satdynamicspatialaptitude,
      title={SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models}, 
      author={Arijit Ray and Jiafei Duan and Ellis Brown and Reuben Tan and Dina Bashkirova and Rose Hendrix and Kiana Ehsani and Aniruddha Kembhavi and Bryan A. Plummer and Ranjay Krishna and Kuo-Hao Zeng and Kate Saenko},
      year={2025},
      eprint={2412.07755},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07755}, 
}
```