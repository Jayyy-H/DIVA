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
# Import Modules
# =========================================================================
from training.diva_utils import DIVAConfig, GatedMLP, CLUB, info_nce_loss, LowRankReadout, orthogonal_loss

logger = get_logger(__name__)

# Original Function Preserved
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

def info_nce(z1, z2, T=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = z1 @ z2.T / T
    labels = torch.arange(z1.size(0), device=z1.device)

    return F.cross_entropy(logits, labels)

def main():
    # -------------------------------------------------------------------------
    # Setup Accelerator & Config (Original Logic)
    # -------------------------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    # Add Stage Argument
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
    # Load Models (Original Logic + DIVA Init)
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

    # Initialize DIVA Modules

    
    diva_cfg = DIVAConfig()
    diva_cfg.stage = args.diva_stage
    
    logger.info(f"===== DIVA Stage {diva_cfg.stage} =====")
    
    # ---------------------------------------------------------
    # Shared Encoders
    # ---------------------------------------------------------

    shared_enc_und = GatedMLP(
        diva_cfg.hidden_dim, diva_cfg.bottleneck_dim
    ).to(accelerator.device)
    
    shared_enc_gen = GatedMLP(
        diva_cfg.hidden_dim, diva_cfg.bottleneck_dim
    ).to(accelerator.device)
    
    # ---------------------------------------------------------
    # Unique Encoders 
    # ---------------------------------------------------------
    unique_enc_und = GatedMLP(
        diva_cfg.hidden_dim, diva_cfg.bottleneck_dim
    ).to(accelerator.device)
    
    unique_enc_gen = GatedMLP(
        diva_cfg.hidden_dim, diva_cfg.bottleneck_dim
    ).to(accelerator.device)
    
    # ---------------------------------------------------------
    # Low-Rank Readouts
    # ---------------------------------------------------------
    A_und = LowRankReadout(
        diva_cfg.bottleneck_dim,
        model.config.vocab_size,
        rank=64,
    ).to(accelerator.device)
    
    A_gen = LowRankReadout(
        diva_cfg.bottleneck_dim,
        model.config.vocab_size,
        rank=64,
    ).to(accelerator.device)
    
   
    club_und = CLUB(
        diva_cfg.bottleneck_dim,
        diva_cfg.bottleneck_dim,
    ).to(accelerator.device)
    
    club_gen = CLUB(
        diva_cfg.bottleneck_dim,
        diva_cfg.bottleneck_dim,
    ).to(accelerator.device)
    
    # =========================================================
    # FREEZE / UNFREEZE STRATEGY
    # =========================================================
    
    if diva_cfg.stage == 1:
        logger.info("Stage 1: Freeze backbone")
        for p in model.parameters():
            p.requires_grad = False
    
    elif diva_cfg.stage == 2:
        logger.info("Stage 2: Joint training")
        for p in model.parameters():
            p.requires_grad = True
    
    # =========================================================
    # OPTIMIZERS
    # =========================================================
    
    main_params = []
    
    if diva_cfg.stage == 2:
        main_params += list(model.parameters())
    
    main_params += (
        list(shared_enc_und.parameters())
        + list(shared_enc_gen.parameters())
        + list(A_und.parameters())
        + list(A_gen.parameters())
    )
    
    if diva_cfg.stage == 2:
        main_params += (
            list(unique_enc_und.parameters())
            + list(unique_enc_gen.parameters())
        )
    
    optimizer = AdamW(
        main_params,
        lr=config.optimizer.params.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.optimizer.params.weight_decay,
    )

    # -------------------------------------------------------------------------
    # Critic optimizers 
    # -------------------------------------------------------------------------

    if diva_cfg.stage == 2:
        optimizer_club_und = AdamW(
            club_und.parameters(),
            lr=diva_cfg.lr_club,
        )
    
        optimizer_club_gen = AdamW(
            club_gen.parameters(),
            lr=diva_cfg.lr_club,
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

    # Prepare with Accelerator 
    if diva_cfg.stage == 2:
    (
        model,
        shared_enc_und, shared_enc_gen,
        unique_enc_und, unique_enc_gen,
        club_und, club_gen,
        optimizer,
        optimizer_club_und, optimizer_club_gen,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        shared_enc_und, shared_enc_gen,
        unique_enc_und, unique_enc_gen,
        club_und, club_gen,
        optimizer,
        optimizer_club_und, optimizer_club_gen,
        train_dataloader,
        lr_scheduler,
    )
    else:
        (
            model,
            shared_enc_und, shared_enc_gen,
            unique_enc_und, unique_enc_gen,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model,
            shared_enc_und, shared_enc_gen,
            unique_enc_und, unique_enc_gen,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    
    mask_scheduler = get_mask_chedule(config.training.noise_type)

    # -------------------------------------------------------------------------
    # Dataset & Dataloader 
    # -------------------------------------------------------------------------
    
    dataset = Text2ImageDataset(
        config.data.dataset.train_data_path,
        tokenizer=tokenizer,
        image_size=config.data.preprocessing.resolution,
        max_seq_length=config.data.preprocessing.max_seq_length,
    )
    
    
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
    # Scheduler 
    # -------------------------------------------------------------------------
    # Calculate total training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    
   
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

    # Recalculate steps after prepare 
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    if not overrode_max_train_steps:
        max_train_steps = config.experiment.max_train_examples_t2i // config.training.batch_size_t2i
    
    
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # -------------------------------------------------------------------------
    # Utils for Prompting & Masking 
    # -------------------------------------------------------------------------
    
    uni_prompting = UniversalPrompting(
        tokenizer, 
        max_text_len=config.data.preprocessing.max_seq_length, 
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>", "<|pad|>"),
        ignore_id=-100, 
        cond_dropout_rate=config.training.cond_dropout_rate,
    )
    
    mask_scheduler = get_mask_chedule(config.training.noise_type)

    # -------------------------------------------------------------------------
    # Resume from Checkpoint 
    # -------------------------------------------------------------------------
    if config.experiment.resume_from_checkpoint:
        if config.experiment.resume_from_checkpoint != "latest":
            path = os.path.basename(config.experiment.resume_from_checkpoint)
        else:
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
    # Training Loop Start
    
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
    
    if diva_cfg.stage == 1:
        model.eval() 
    
    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # ==========================================
                # A. Prepare Inputs (Shared Anchor)
                # ==========================================
                # batch contains: 'pixel_values', 'input_ids' (caption), 'attention_mask'
                with torch.no_grad():
                    image_tokens = vq_model.get_codebook_indices(
                            batch["pixel_values"].to(dtype=model.dtype)
                    )
                
                input_ids = batch["input_ids"]
                
                # ---------------------------------------------------------
                # 2. Generation Flow (t2i)
                # ---------------------------------------------------------
                ret_gen = model(
                    input_ids=input_ids,
                    image_tokens=image_tokens,
                    input_type="t2i",
                    mask_scheduler=mask_scheduler,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                # middle-layer features → [B, hidden_dim]
                feat_gen = ret_gen.hidden_states[
                    diva_cfg.middle_layer_idx
                ].mean(dim=1)
                
                # ---------------------------------------------------------
                # 3. Understanding Flow (mmu)
                # ---------------------------------------------------------
                ret_und = model(
                    input_ids=input_ids,
                    image_tokens=image_tokens,
                    input_type="mmu",
                    labels=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                feat_und = ret_und.hidden_states[
                    diva_cfg.middle_layer_idx
                ].mean(dim=1)
                
                # =========================================================
                # 4. Factorization
                # =========================================================
                
                # -------- Shared representations --------
                z_sh_gen = shared_enc_gen(feat_gen)
                z_sh_und = shared_enc_und(feat_und)
                
                # -------- Unique representations --------
                if diva_cfg.stage == 1:
                    # Stage 1: no unique branch
                    z_un_gen = torch.zeros_like(z_sh_gen)
                    z_un_und = torch.zeros_like(z_sh_und)
                else:
                    z_un_gen = unique_enc_gen(feat_gen)
                    z_un_und = unique_enc_und(feat_und)

                
                if diva_cfg.stage == 2:

                    # -----------------------------------------------------
                    # Critic Update 
                    # -----------------------------------------------------
                    optimizer_club_und.zero_grad()
                    optimizer_club_gen.zero_grad()
                
                    # Detach 
                    loss_club_und = club_und.loglikeli(
                        z_sh_und.detach(),
                        z_un_und.detach(),
                    )
                
                    loss_club_gen = club_gen.loglikeli(
                        z_sh_gen.detach(),
                        z_un_gen.detach(),
                    )
                
                    loss_club_total = -(loss_club_und + loss_club_gen)
                
                    accelerator.backward(loss_club_total)
                
                    optimizer_club_und.step()
                    optimizer_club_gen.step()
                
                    # -----------------------------------------------------
                    # MI Estimation 
                    # -----------------------------------------------------
                    
                    for p in club_und.parameters():
                        p.requires_grad = False
                    for p in club_gen.parameters():
                        p.requires_grad = False
                
                    loss_mi_und = club_und.mi_est(z_sh_und, z_un_und)
                    loss_mi_gen = club_gen.mi_est(z_sh_gen, z_un_gen)
                
                    loss_mi = loss_mi_und + loss_mi_gen
                
                   
                    for p in club_und.parameters():
                        p.requires_grad = True
                    for p in club_gen.parameters():
                        p.requires_grad = True
                
                else:
                    loss_mi = torch.tensor(0.0, device=z_sh_gen.device)
                
                logits_gen = ret_gen.logits
                
                logits_und = ret_und.logits
                
                if diva_cfg.stage == 1:
                    bias_gen = A_gen(z_sh_gen)
                    bias_und = A_und(z_sh_und)
                else:
                    bias_gen = A_gen(z_sh_gen + z_un_gen)
                    bias_und = A_und(z_sh_und + z_un_und)
                
                
                bias_gen = bias_gen.unsqueeze(1)
                bias_und = bias_und.unsqueeze(1)
                
                logits_gen = logits_gen + bias_gen
                logits_und = logits_und + bias_und
                
                
                # ---------------------------------------------------------
                # Recompute Cross Entropy
                # ---------------------------------------------------------
                
                loss_gen = F.cross_entropy(
                    logits_gen.view(-1, logits_gen.size(-1)),
                    ret_gen.labels.view(-1),
                    ignore_index=-100,
                )
                
                loss_und = F.cross_entropy(
                    logits_und.view(-1, logits_und.size(-1)),
                    ret_und.labels.view(-1),
                    ignore_index=-100,
                )
                
                
                # ---------------------------------------------------------
                # Shared Alignment 
                # ---------------------------------------------------------
                
                loss_align = info_nce_loss(
                    z_sh_und,
                    z_sh_gen,
                    temperature=diva_cfg.temp,
                )
                
                
                # ---------------------------------------------------------
                # Orthogonality (Stage 2 only)
                # ---------------------------------------------------------
                
                if diva_cfg.stage == 2:
                    loss_orth = (
                        orthogonal_loss(z_sh_und, z_un_und)
                        + orthogonal_loss(z_sh_gen, z_un_gen)
                    )
                else:
                    loss_orth = torch.tensor(0.0, device=loss_gen.device)
                
                
                # ---------------------------------------------------------
                # Total Loss
                # ---------------------------------------------------------
                
                if diva_cfg.stage == 1:
                
                    loss_total = (
                        loss_gen
                        + loss_und
                        + diva_cfg.lambda_sha * loss_align
                    )
                
                else:
                
                    loss_total = (
                        loss_gen
                        + loss_und
                        + diva_cfg.lambda_sha * loss_align
                        + diva_cfg.lambda_uni * loss_mi
                        + diva_cfg.lambda_orth * loss_orth
                    )
                
                
                optimizer.zero_grad()
                
                accelerator.backward(loss_total)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        main_params,
                        config.training.max_grad_norm,
                    )
                
                optimizer.step()
                lr_scheduler.step()

            # -------------------------------------------------------------------------
            # Logging 
            # -------------------------------------------------------------------------
            if accelerator.sync_gradients:
                if global_step % config.experiment.log_every == 0:
                    # Calculate avg loss across processes for logging
                    avg_loss = accelerator.gather(loss_total.repeat(config.training.batch_size_t2i)).mean()
                    
                    wandb.log({k: v for k,v in locals().items() if k.startswith("loss_")})

                    
                    if accelerator.is_main_process:
                        wandb.log({
                            "train/total_loss": avg_loss.item(),
                            "train/loss_gen": loss_gen.item(),
                            "train/loss_und": loss_und.item(),
                            "train/loss_align": loss_align.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        })

            # -------------------------------------------------------------------------
            # Checkpointing (Original Logic Preserved)
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
                    torch.save(accelerator.unwrap_model(club_und).state_dict(), save_path / "diva_club_und.pth")
                    torch.save(accelerator.unwrap_model(club_gen).state_dict(), save_path / "diva_club_gen.pth")

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
