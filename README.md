# SAT: Spatial Aptitude Training for Multimodal Language Models
Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, Kate Saenko

[Project Page](https://arijitray1993.github.io/SAT/)
[Paper](https://arxiv.org/abs/2412.07755)

![SAT Data](https://arijitray1993.github.io/SAT/SAT_webpage/static/images/sat_teaser.png)


## Pre-requisities

Create a conda environment:

```
conda env create -f sat_conda.yml
conda activate sat_conda
```

## Run Inference with our best SAT model

```python
import models.model_interface as models

from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
from huggingface_hub import hf_hub_download

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
```

## Get the Data
Follow instructions here: https://huggingface.co/datasets/array/SAT 


## Training
Coming soon.


## BibTeX

If you use this code/data, please cite:

```
@misc{ray2024satspatialaptitudetraining,
      title={SAT: Spatial Aptitude Training for Multimodal Language Models}, 
      author={Arijit Ray and Jiafei Duan and Reuben Tan and Dina Bashkirova and Rose Hendrix and Kiana Ehsani and Aniruddha Kembhavi and Bryan A. Plummer and Ranjay Krishna and Kuo-Hao Zeng and Kate Saenko},
      year={2024},
      eprint={2412.07755},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07755}, 
}
```