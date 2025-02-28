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
lora_model = PeftModel.from_pretrained(load_model, "array/sat-dynamic-13b")
lora_model = lora_model.half()
model = lora_model.merge_and_unload()


## Process the data
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