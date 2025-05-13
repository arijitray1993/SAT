import os
import pdb  # noqa
import sys  # noqa
import math
# import time
from collections import defaultdict  # noqa
 
# training related imports
from models import eval_funcs as eval
import hydra
import custom_datasets.dataloaders as datasets
import models.model_interface as models
import torch
import tqdm

from accelerate import Accelerator
from accelerate import PartialState
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, BitsAndBytesConfig
from safetensors import safe_open
 
# import transformers
 
# import models.preprocess as preprocess
import wandb  # noqa
from omegaconf import DictConfig, OmegaConf  # noqa
 
# utils
from utils import utils
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
from peft import prepare_model_for_kbit_training
import gc
import time
import numpy as np
import random
from omegaconf import OmegaConf,open_dict

# fix all seeds
seed = 20249
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)

class GenericModule():
    def __init__(
        self,
        cfg,
        model,
        model_input_choice,
        train_dataloader, 
        test_dataloader,
        metrics=None,
        lr=None,
        max_epochs=0,
        weight_decay=0.01,
        lr_scheduler_choice=None,
        load_checkpoint=None,
        eval_every = 100,
        logger = None, # expects a wandb run object
        exp_name = "",
        eval_only = False,
        num_eval_steps = 5,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.logger = logger
        self.exp_name = exp_name
        self.eval_only = eval_only
        self.num_eval_steps = num_eval_steps

        # initialize eval functions and metrics
        self.metrics = metrics
        for metric_entry in metrics:
            metric_name = metric_entry[0]
            metric_func = metric_entry[1]
            # metric_args = metric_entry[2]
            # print(metric_name)
            metric_args = {}
            metric_args["exp_name"] = self.exp_name
            metric_args["logger"] = self.logger
            setattr(self, metric_name, getattr(eval, metric_func)(metric_args))

        self.metric_to_track = []

        # set up training
        num_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
        self.accelerator = Accelerator(gradient_accumulation_steps=num_accumulation_steps)
        if train_dataloader is not None:
            self.num_epochs = max_epochs
            self.num_training_steps = self.num_epochs * len(train_dataloader)

            num_warmup_steps = cfg.get("num_warmup_steps", 0)

            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.lr_scheduler = get_scheduler(
                lr_scheduler_choice,
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.num_training_steps
            )

            self.accelerator.register_for_checkpointing(self.lr_scheduler)

        if load_checkpoint:
            self.accelerator.load_state(load_checkpoint)

        self.model_input_choice = model_input_choice
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        if train_dataloader is not None:
            self.train_dataloader, self.test_dataloader, self.model, self.optimizer = self.accelerator.prepare(
                self.train_dataloader, self.test_dataloader, self.model, self.optimizer)
        else:
            self.test_dataloader, self.model = self.accelerator.prepare(
                self.test_dataloader, self.model)
        self.eval_every = eval_every
        # self.grad_acc_step = 1
        

    def train(self,):
        progress_bar = tqdm.tqdm(range(self.num_training_steps))
        self.global_step = 0
        for epoch in range(self.num_epochs):
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    #outputs = self.model.forward(*[batch[inp] for inp in self.model_input_choice])
                    # try:
                        # pdb.set_trace()
                    # pdb.set_trace()
                    outputs = self.model.forward(**{inp: batch[inp] for inp in self.model_input_choice})
                    
                    # except:
                    #   pdb.set_trace()
                    #    print("Error in forward pass. Skipping step...")
                    #    # clear cuda memory
                    #    torch.cuda.empty_cache()
                    #    gc.collect()
                    #    continue
                    loss = outputs.loss
                    
                    if self.cfg.get("use_feature_loss"):
                        # pdb.set_trace()
                        feature_loss = outputs["im_feature_loss"]
                        total_loss = self.cfg.feature_lambda * feature_loss + (1-self.cfg.feature_lambda) * loss
                        loss = total_loss

                    # check if loss is nan
                    if math.isnan(loss.item()):
                        # pdb.set_trace()
                        print("Loss is nan. Skipping step...")
                        self.optimizer.zero_grad()
                        continue

                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    progress_bar.set_description(f"train_loss: {loss.item()}")
                    progress_bar.update(1)
                    self.logger.log({"train_loss": loss.item(), "step": self.global_step})

                    # validate at certain intervals
                    if (self.global_step % self.eval_every == 0) and self.global_step>10000:
                        self.validate()
                    
                        # compute metrics after running all eval loop
                        for entry in self.metrics:
                            metric_name = entry[0]
                            getattr(self, metric_name).compute()

                    self.global_step += 1
 
    def validate(self,):
        progress_bar = tqdm.tqdm(range(len(self.test_dataloader)))
        count = 0
        self.model.eval()
        #distributed_state = PartialState()
        #self.model.to(distributed_state.device)
        for batch in self.test_dataloader:
            #self.model_outputs = self.model.forward(
            #    *[batch[inp] for inp in self.model_input_choice]
            #)
            with torch.no_grad():
                # with distributed_state.split_between_processes(batch) as entry:
                try:
                    generated_ids = self.model.generate(**{inp: batch[inp] for inp in self.model_input_choice})
                except:
                    pdb.set_trace()

            generated_ids[generated_ids==-200] = 1
            generated_text = self.test_dataloader.dataset.batch_decode(generated_ids, skip_special_tokens=True)# [0].strip()
            
            print("Generated text: ")
            print(generated_text)

            print("Ground truth text: ")
            print(batch["text_labels"][0])


            for entry in self.metrics:
                metric_name = entry[0]
                
                getattr(self, metric_name).update(generated_text, batch)
                
                #self.logger.log(
                #    metric_name,
                #    getattr(self, metric_name),
                #)
            progress_bar.set_description(f"")
            progress_bar.update(1)
            count+=1
            if count>self.num_eval_steps:
                break
        
        # compute metrics after running all eval loop
        for entry in self.metrics:
            metric_name = entry[0]
            getattr(self, metric_name).compute()

        if not self.eval_only:
            self.save_model()
        self.model.train()

    def save_model(self,):
        # save the huggingface model locally
        self.model.save_pretrained(f"checkpoints/{self.exp_name}/" +f"{self.logger.name}_{self.global_step}")


@hydra.main(version_base=None, config_path="confs/", config_name="exp_config")
def main(cfg: DictConfig):
    cfg = cfg.exp_name
    print(os.getcwd())

    if not os.path.exists("checkpoints/" + cfg.exp_name):
        os.mkdir(
            "checkpoints/"
            + cfg.exp_name
        )

    wandb.login()
    run = wandb.init(project=cfg.exp_name, config=dict(cfg))

    # =============== define model choice ============================
    if cfg.get("load_checkpoint"):
        model = getattr(models, cfg.model_choice).from_pretrained(cfg.load_checkpoint, args=cfg.model_init_args, local_files_only=True)    
    else:
        model = getattr(models, cfg.model_choice)(cfg.model_init_args)

    # ======================= load dataloaders =====================
    dataloader_train = None
    if not cfg.eval_only:
        train_dataset = getattr(datasets, cfg.train_dataset_choice)(
            cfg.train_dataset_args, model.tokenizer, model.image_processor
        )
        # this chooses the dataset of choice from data/datasets.py
        # according to the variable "train_dataset_choice" in cfg
        # and initializes it with cfg.train_dataset_args
    
        dataloader_train = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            collate_fn=train_dataset.collate_fn
            if cfg.get("train_collate_fn")
            else None,
            shuffle=False if cfg.get("no_shuffle") or cfg.get("use_sampler") else True,
            num_workers = cfg.num_workers,
            sampler=train_dataset.sampler if cfg.get("use_sampler") else None,
        )
    
    if cfg.eval_only:
        # if eval only, load test data, or else
        # load val data while training to choose hyperparameters.
        
        dataset_val = getattr(datasets, cfg.val_dataset_choice)(cfg.val_dataset_args, model.tokenizer, model.image_processor)
    else:
        
        dataset_val = getattr(datasets, cfg.valtrain_dataset_choice)(
            cfg.valtrain_dataset_args, model.tokenizer, model.image_processor
        )
    
    val_batch_size=cfg.get("val_batch_size", 1)

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        collate_fn=dataset_val.collate_fn if cfg.get("val_collate_fn") else None,
        num_workers = cfg.num_workers,
        shuffle=False
    )
    print("data loaded...")
    
    # =========== Define training choices ===========================
    if cfg.eval_only:
        model.eval()  # turns batch norm off.
        model.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False
            
    
    # pdb.set_trace()
    if cfg.get("tune_lora"):
        if cfg.get("freeze_vision"):
            cls_l = torch.nn.Linear
            lora_module_names = set()
            multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
            for name, module in model.named_modules():
                if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                    continue
                if isinstance(module, cls_l):
                    #names = name.split('.')
                    lora_module_names.add(name)

            if 'lm_head' in lora_module_names: # needed for 16-bit
                lora_module_names.remove('lm_head')
            lora_module_names = list(lora_module_names)
        else:
            lora_module_names = ["q_proj", "v_proj"]
            if 'instructblip' in cfg.model_choice.lower():
                lora_module_names = ["q_proj", "v_proj", "query", "key", "value", "qkv"]

        # pdb.set_trace()
        lora_rank = cfg.get("lora_rank", 128)
        lora_alpha = cfg.get("lora_alpha", 256)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_module_names,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=[],
        )

        #model.model.gradient_checkpointing_enable()
        #model.model = prepare_model_for_kbit_training(model.model)

        if cfg.get("lora_model_path"):
            lora_model = PeftModel.from_pretrained(model, cfg.get("lora_model_path"))
            
            if not cfg.eval_only:
                # go through the dict and set lora layers to requires grad True
                lora_model = lora_model.half()
                for name, param in lora_model.named_parameters():
                    set_grad =  ("q_proj" in name or "v_proj" in name) and ('weight' in name or 'bias' in name) and ('default' in name)
                    for module in lora_config.modules_to_save:
                        if module in name:
                            set_grad = True
                    if set_grad:
                        # pdb.set_trace()
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for name, param in lora_model.named_parameters():
                    param.requires_grad = False
                    
                lora_model = lora_model.merge_and_unload()
                lora_model = lora_model.half()
                

        else:
            lora_model = get_peft_model(model, lora_config)
            lora_model = lora_model.half()

        # pdb.set_trace()
        print("Lora models loaded...")
        
        #if cfg.get('offload_vision'):
        #    lora_model.model.model.model.vision_tower.cpu()


        for name, param in lora_model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

        total_params = sum(param.numel() for param in lora_model.parameters())
        print(f"Total number of params: {total_params}")

        total_params = sum(param.numel() for param in lora_model.parameters() if param.requires_grad)
        print(f"    Trainable params: {total_params}")

        # pdb.set_trace()
        
        # === training module ===
        training_module = GenericModule(
            cfg=cfg,
            model=lora_model,
            model_input_choice=cfg.model_input_choice,
            metrics=cfg.metrics,
            lr=cfg.lr,
            max_epochs=cfg.num_epochs,
            weight_decay=cfg.weight_decay,
            lr_scheduler_choice=cfg.get("lr_scheduler"),
            load_checkpoint=cfg.load_checkpoint,
            train_dataloader=dataloader_train,
            test_dataloader=dataloader_val,
            eval_every=cfg.eval_every,
            logger = run,
            exp_name = cfg.exp_name,
            eval_only = cfg.eval_only,
            num_eval_steps = cfg.num_eval_steps,
        )

    else:
        print("models loaded...")
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

        total_params = sum(param.numel() for param in model.parameters())
        print(f"Total number of params: {total_params}")

        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"    Trainable params: {total_params}")

        # === training module ===
        training_module = GenericModule(
            cfg=cfg,
            model=model,
            model_input_choice=cfg.model_input_choice,
            metrics=cfg.metrics,
            lr=cfg.lr,
            max_epochs=cfg.num_epochs,
            weight_decay=cfg.weight_decay,
            lr_scheduler_choice=cfg.get("lr_scheduler"),
            load_checkpoint=cfg.load_checkpoint,
            train_dataloader=dataloader_train,
            test_dataloader=dataloader_val,
            eval_every=cfg.eval_every,
            logger = run,
            exp_name = cfg.exp_name,
            eval_only = cfg.eval_only,
            num_eval_steps = cfg.num_eval_steps,
        )

    if cfg.eval_only:
        training_module.validate()
    else:
        training_module.train()

    wandb.finish()


if __name__ == "__main__":

   main()
