# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset
# Retain original dataset imports to avoid breaking dependencies
from training.imagenet_dataset import ImageNetDataset 
from parquet import RefinedWebDataset

from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, \
    create_attention_mask_for_mmu
from models.lr_schedulers import get_scheduler

# =========================================================================
# [DIVA] Import Custom Modules
# =========================================================================
from training.diva_utils import DIVAConfig, GatedMLP, CLUB, info_nce_loss

logger = get_logger(__name__)

# [Original Helper Function Preserved]
def get_grouped_params(model, weight_decay, no_decay_name_list=["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight", "ln_k.weight", "ln_q.weight", "ln_v.weight", "ln_1.weight", "ln_2.weight"]):
    params_with_decay = []
    params_with_decay_names = []
    params_without_decay = []
    params_without_decay_names = []
    for n, p in model.named_parameters():
        if any(t in n for t in no_decay_name_list):
            params_without_decay.append(p)
            params_without_decay_names.append(n)
        else:
            params_with_decay.append(p)
            params_with_decay_names.append(n)
    return [
        {"params": params_with_decay, "weight_decay": weight_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ]

def main():
    # -------------------------------------------------------------------------
    # 1. Setup Accelerator & Config (Original Logic)
    # -------------------------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    # [DIVA] Add Stage Argument
    parser.add_argument("--diva_stage", type=int, default=2, help="1 for Decomposition, 2 for Mutual Reinforcement")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    if accelerator.is_local_main_process:
        wandb.init(
            project=config.experiment.project,
            name=f"{config.experiment.name}-stage{args.diva_stage}",
            config=OmegaConf.to_container(config),
        )

    set_seed(42)

    # -------------------------------------------------------------------------
    # 2. Load Models (Original Logic + DIVA Init)
    # -------------------------------------------------------------------------
    # VQ Model
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    for param in vq_model.parameters():
        param.requires_grad = False

    # Show-o Model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
    else:
        model = Showo(**config.model.showo).to(accelerator.device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path, 
        padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # [DIVA] Initialize DIVA Modules
    diva_cfg = DIVAConfig()
    diva_cfg.stage = args.diva_stage # Override config with CLI arg
    
    logger.info(f"*** DIVA Initialization: Running Stage {diva_cfg.stage} ***")

    # Encoders (Shared & Unique)
    shared_enc_und = GatedMLP(diva_cfg.hidden_dim, diva_cfg.hidden_dim).to(accelerator.device)
    shared_enc_gen = GatedMLP(diva_cfg.hidden_dim, diva_cfg.hidden_dim).to(accelerator.device)
    unique_enc_und = GatedMLP(diva_cfg.hidden_dim, diva_cfg.hidden_dim).to(accelerator.device)
    unique_enc_gen = GatedMLP(diva_cfg.hidden_dim, diva_cfg.hidden_dim).to(accelerator.device)
    
    # CLUB Discriminator
    club_disc = CLUB(diva_cfg.hidden_dim, diva_cfg.hidden_dim).to(accelerator.device)

    # -------------------------------------------------------------------------
    # 3. Two-Stage Optimizer Configuration (Crucial Logic)
    # -------------------------------------------------------------------------
    
    # Default weight decay logic from original code
    weight_decay = config.optimizer.params.weight_decay
    
    # Parameter Grouping Logic
    optimizer_grouped_parameters = []

    # [DIVA Stage 1]: Decomposition
    # FREEZE Backbone, Train ONLY Encoders.
    if diva_cfg.stage == 1:
        logger.info("[DIVA Stage 1] Freezing Show-o Backbone. Training only DIVA Encoders.")
        for param in model.parameters():
            param.requires_grad = False
            
        # Only add DIVA params to optimizer
        diva_params = list(shared_enc_und.parameters()) + list(shared_enc_gen.parameters()) + \
                      list(unique_enc_und.parameters()) + list(unique_enc_gen.parameters())
        
        optimizer_grouped_parameters = [{"params": diva_params, "weight_decay": weight_decay}]

    # [DIVA Stage 2]: Mutual Reinforcement
    # UNFREEZE Backbone (or LoRA), Train Everything.
    elif diva_cfg.stage == 2:
        logger.info("[DIVA Stage 2] Unfreezing Show-o Backbone. Joint Training enabled.")
        for param in model.parameters():
            param.requires_grad = True # Or use LoRA config here if needed
            
        # 1. Backbone Params (with careful decay grouping)
        backbone_groups = get_grouped_params(model, weight_decay)
        optimizer_grouped_parameters.extend(backbone_groups)
        
        # 2. DIVA Encoder Params
        diva_params = list(shared_enc_und.parameters()) + list(shared_enc_gen.parameters()) + \
                      list(unique_enc_und.parameters()) + list(unique_enc_gen.parameters())
        optimizer_grouped_parameters.append({"params": diva_params, "weight_decay": weight_decay})

    # Main Optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.optimizer.params.learning_rate,
        betas=(config.optimizer.params.beta1, config.optimizer.params.beta2),
        eps=config.optimizer.params.epsilon,
    )
    
    # CLUB Optimizer (Always separate, used for maximization step)
    optimizer_club = AdamW(club_disc.parameters(), lr=diva_cfg.lr_club)

    # -------------------------------------------------------------------------
    # 4. Dataset (Focusing on T2I for Pairs)
    # -------------------------------------------------------------------------
    # Using original Text2ImageDataset logic
    dataset = Text2ImageDataset(
        config.data.dataset.train_data_path,
        tokenizer=tokenizer,
        image_size=config.data.preprocessing.resolution,
        max_seq_length=config.data.preprocessing.max_seq_length,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size_t2i,
        shuffle=True,
        num_workers=config.data.dataset.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Schedulers (Original Logic)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
    max_train_steps = config.experiment.max_train_examples_t2i // config.training.batch_size_t2i
    
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Prepare with Accelerator (Crucial for Distributed Training)
    (
        model, 
        shared_enc_und, shared_enc_gen, unique_enc_und, unique_enc_gen, club_disc,
        optimizer, optimizer_club, dataloader, lr_scheduler
    ) = accelerator.prepare(
        model, 
        shared_enc_und, shared_enc_gen, unique_enc_und, unique_enc_gen, club_disc,
        optimizer, optimizer_club, dataloader, lr_scheduler
    )

    # Helper Utils
    mask_scheduler = get_mask_chedule(config.training.noise_type)

    # 4. Dataset & Dataloader (Original Logic Preserved)
    # -------------------------------------------------------------------------
    # [DIVA Note]: We focus on Text2ImageDataset because it provides paired (Image, Text).
    # This acts as our "Semantic Anchor" source.
    dataset = Text2ImageDataset(
        config.data.dataset.train_data_path,
        tokenizer=tokenizer,
        image_size=config.data.preprocessing.resolution,
        max_seq_length=config.data.preprocessing.max_seq_length,
    )
    
    # Original logic uses CombinedLoader if multiple datasets exist.
    # For DIVA Post-training, we simplify to single loader to ensure strict pairing.
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size_t2i,
        shuffle=True,
        num_workers=config.data.dataset.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.data.dataset.persistent_workers,
    )

    # -------------------------------------------------------------------------
    # 5. Scheduler & Math (Original Logic Preserved)
    # -------------------------------------------------------------------------
    # Calculate total training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    
    # Using T2I examples count as the main counter
    if config.experiment.max_train_examples_t2i is None:
        max_train_steps = config.experiment.max_train_steps
        overrode_max_train_steps = True
    else:
        max_train_steps = config.experiment.max_train_examples_t2i // config.training.batch_size_t2i

    # Initialize Scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # -------------------------------------------------------------------------
    # 6. Accelerator Prepare (Critical for Distributed Training)
   
    (
        model, 
        shared_enc_und, shared_enc_gen, unique_enc_und, unique_enc_gen, club_disc,
        optimizer, optimizer_club, train_dataloader, lr_scheduler
    ) = accelerator.prepare(
        model, 
        shared_enc_und, shared_enc_gen, unique_enc_und, unique_enc_gen, club_disc,
        optimizer, optimizer_club, train_dataloader, lr_scheduler
    )

    # Recalculate steps after prepare (for distributed adjustments)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    if not overrode_max_train_steps:
        max_train_steps = config.experiment.max_train_examples_t2i // config.training.batch_size_t2i
    
    # We need to calculate the number of epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # -------------------------------------------------------------------------
    # 7. Utils for Prompting & Masking (Original Logic Preserved)
    # -------------------------------------------------------------------------
    # Universal Prompting is needed to construct input_ids for both T2I and MMU tasks
    uni_prompting = UniversalPrompting(
        tokenizer, 
        max_text_len=config.data.preprocessing.max_seq_length, 
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>", "<|pad|>"),
        ignore_id=-100, 
        cond_dropout_rate=config.training.cond_dropout_rate,
    )
    
    mask_scheduler = get_mask_chedule(config.training.noise_type)

    # -------------------------------------------------------------------------
    # 8. Resume from Checkpoint (Original Logic Preserved)
    # -------------------------------------------------------------------------
    if config.experiment.resume_from_checkpoint:
        if config.experiment.resume_from_checkpoint != "latest":
            path = os.path.basename(config.experiment.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(config.experiment.output_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.experiment.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.experiment.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.experiment.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
        first_epoch = 0

    # -------------------------------------------------------------------------
    # 9. Training Loop Start
    # -------------------------------------------------------------------------
    total_batch_size = config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps

    logger.info("***** Running DIVA Post-Training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_t2i}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Current DIVA Stage = {diva_cfg.stage}")

    progress_bar = None
    if accelerator.is_local_main_process:
        from tqdm.auto import tqdm
        progress_bar = tqdm(range(max_train_steps), initial=initial_global_step, desc="Training")

    global_step = initial_global_step

    # Set model to train
    model.train()
    
    # [DIVA Stage 1 Check]: Ensure backbone is strictly eval mode if needed, though optimizer handles grads.
    # But setting eval() might disable dropout which is desired for freezing.
    if diva_cfg.stage == 1:
        model.eval() 
    
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # ==========================================
                # A. Prepare Inputs (Shared Anchor)
                # ==========================================
                # batch contains: 'pixel_values', 'input_ids' (caption), 'attention_mask'
                pixel_values = batch['pixel_values'].to(dtype=model.dtype)
                input_ids = batch['input_ids']
                
                # 1. Encode Images to Discrete Tokens (VQ-VAE)
                # We do this once, shared for both streams.
                with torch.no_grad():
                    image_tokens = vq_model.get_codebook_indices(pixel_values) # [B, 1024] or similar

                # ==========================================
                # B. Stream 1: Generation (Text-to-Image)
                # ==========================================
                # Show-o Input: <t2i> text <soi> [MASK] <eoi>
                # We pass input_ids (text) and image_tokens (target). 
                # Model handles masking internally via mask_scheduler.
                
                ret_gen = model(
                    input_ids=input_ids,
                    image_tokens=image_tokens,
                    input_type="t2i",
                    mask_scheduler=mask_scheduler,
                    output_hidden_states=True, # [DIVA Required]
                    return_dict=True
                )
                loss_gen = ret_gen.loss
                
                # Extract Middle Layer Features for Generation
                # Shape: [Batch, Seq, Dim]. We pool to get [Batch, Dim]
                # Note: We assume Layer 16 (or configured layer). 
                # We perform mean pooling to get a robust representation.
                feat_gen_raw = ret_gen.hidden_states[diva_cfg.middle_layer_idx].mean(dim=1)

                # ==========================================
                # C. Stream 2: Understanding (MMU / Captioning)
                # ==========================================
                # Show-o Input: <mmu> <soi> image <eoi> <sov> text <eov>
                # We use the SAME image and text. Task: Predict text given image.
                
                ret_und = model(
                    input_ids=input_ids,
                    image_tokens=image_tokens,
                    input_type="mmu",
                    labels=input_ids, # Standard Causal LM training (Text is target)
                    output_hidden_states=True, # [DIVA Required]
                    return_dict=True
                )
                loss_und = ret_und.loss
                
                # Extract Middle Layer Features for Understanding
                feat_und_raw = ret_und.hidden_states[diva_cfg.middle_layer_idx].mean(dim=1)

                # ==========================================
                # D. DIVA Logic: Decomposition & Loss
                # ==========================================
                
                # 1. Project to Shared/Unique Spaces (Decomposition)
                z_sh_und = shared_enc_und(feat_und_raw)
                z_sh_gen = shared_enc_gen(feat_gen_raw)
                
                z_un_und = unique_enc_und(feat_und_raw)
                z_un_gen = unique_enc_gen(feat_gen_raw)
                
                # 2. CLUB Optimization (Maximization Step)
                # We must train the discriminator to estimate p(y|x) well.
                # Crucial: Detach input features so we don't backprop to encoders yet.
                loss_club_update = club_disc.loglikeli(z_sh_und.detach(), z_un_und.detach()) + \
                                   club_disc.loglikeli(z_sh_gen.detach(), z_un_gen.detach())
                
                # Backward for CLUB Discriminator
                # Note: We want to Maximize LogLikelihood, so Minimize -LogLikelihood
                optimizer_club.zero_grad()
                accelerator.backward(-loss_club_update) 
                optimizer_club.step()
                
                # 3. DIVA Regularization Losses (Minimization Step)
                
                # (a) Alignment (InfoNCE): Pull shared features together
                loss_align = info_nce_loss(z_sh_und, z_sh_gen, temperature=diva_cfg.temp)
                
                # (b) Disentanglement (CLUB MI Upper Bound): Minimize MI between Shared and Unique
                # This pushes Shared and Unique to be independent.
                mi_und = club_disc.mi_est(z_sh_und, z_un_und)
                mi_gen = club_disc.mi_est(z_sh_gen, z_un_gen)
                loss_dis = (mi_und + mi_gen) / 2
                
                # ==========================================
                # E. Total Loss Calculation & Backward
                # ==========================================
                # Weighted Sum
                # If Stage 1: Backbone is frozen, loss_gen/loss_und gradients won't update backbone.
                #             But DIVA gradients will update Encoders.
                # If Stage 2: All updates.
                
                loss_total = loss_gen + loss_und + \
                             diva_cfg.lambda_align * loss_align + \
                             diva_cfg.lambda_dis * loss_dis

                accelerator.backward(loss_total)
                
                # Gradient Clipping
                if accelerator.sync_gradients:
                    # Clip gradients for all trainable parameters
                    params_to_clip = list(model.parameters()) + \
                                     list(shared_enc_und.parameters()) + list(shared_enc_gen.parameters()) + \
                                     list(unique_enc_und.parameters()) + list(unique_enc_gen.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # -------------------------------------------------------------------------
            # 11. Logging (Original Logic + DIVA Metrics)
            # -------------------------------------------------------------------------
            if accelerator.sync_gradients:
                if global_step % config.experiment.log_every == 0:
                    # Calculate avg loss across processes for logging
                    avg_loss = accelerator.gather(loss_total.repeat(config.training.batch_size_t2i)).mean()
                    
                    logger.info(
                        f"Step {global_step}: Total={avg_loss.item():.4f} | "
                        f"Gen={loss_gen.item():.4f} | Und={loss_und.item():.4f} | "
                        f"Align={loss_align.item():.4f} | MI={loss_dis.item():.4f}"
                    )
                    
                    if accelerator.is_main_process:
                        wandb.log({
                            "train/total_loss": avg_loss.item(),
                            "train/loss_gen": loss_gen.item(),
                            "train/loss_und": loss_und.item(),
                            "train/loss_align": loss_align.item(),
                            "train/loss_dis": loss_dis.item(),
                            "train/club_loglikeli": loss_club_update.item(), # Monitor discriminator quality
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        })

            # -------------------------------------------------------------------------
            # 12. Checkpointing (Original Logic Preserved)
            # -------------------------------------------------------------------------
            if global_step % config.experiment.save_every == 0 and global_step > 0:
                save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}"
                
                # Save Model Backbone
                # Logic to handle DeepSpeed/FSDP unwrapping
                state_dict = accelerator.get_state_dict(model)
                
                if accelerator.is_main_process:
                    # Save Backbone
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        save_path / "unwrapped_model",
                        save_function=accelerator.save,
                        state_dict=state_dict,
                        safe_serialization=False
                    )
                  
                    torch.save(accelerator.unwrap_model(shared_enc_und).state_dict(), save_path / "diva_shared_und.pth")
                    torch.save(accelerator.unwrap_model(shared_enc_gen).state_dict(), save_path / "diva_shared_gen.pth")
                    torch.save(accelerator.unwrap_model(unique_enc_und).state_dict(), save_path / "diva_unique_und.pth")
                    torch.save(accelerator.unwrap_model(unique_enc_gen).state_dict(), save_path / "diva_unique_gen.pth")
                    torch.save(accelerator.unwrap_model(club_disc).state_dict(), save_path / "diva_club.pth")

                    json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
                    logger.info(f"Saved state to {save_path}")

            global_step += 1
            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break

    logger.info("DIVA Post-Training Finished.")
    accelerator.end_training()

if __name__ == "__main__":
    main()
