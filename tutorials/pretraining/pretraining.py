import argparse
import ast
import copy
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from methylgpt.model.methyl_datasets import create_dataloader
from methylgpt.model.methyl_model import MethylGPTModel
from methylgpt.model.methyl_vocab import MethylVocab
from methylgpt.model.methyl_loss import masked_mse_loss
from methylgpt.utils.logging import setup_logger, add_console_handler
# from methylgpt.utils.plot_embeddings import plot_umap_categorical, plot_umap_numerical # Not used in current main loop
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value  # Ensure methylgpt is imported before to initialize paths
from utils import set_seed, split_files, save_config, make_hash

try:
    # Auto Mixed Precision
    from torch.cuda.amp import GradScaler
    amp_available = torch.cuda.is_available()
    if not amp_available:
        warnings.warn("torch.cuda.amp.GradScaler imported but CUDA is not available. AMP will be disabled.")
except ImportError:
    amp_available = False
    warnings.warn("torch.cuda.amp.GradScaler not available. AMP will be disabled.")

try:
    from flash_attn.flash_attention import FlashMHA
    flash_attn_available = True
except ImportError:
    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

# ------------------------ Config & Setup ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-config_file", "--config_file", default="config_example.json",
                    help="Path to the config file")
parser.add_argument("-savename", "--savename", default="pretrain_run",
                    help="Name for saving outputs")
parser.add_argument("-probe_id_path", "--probe_id_path", default="probe_ids_type3.csv",
                    help="Path to probe IDs CSV file for VOCABULARY creation")
parser.add_argument("-parquet_data_dir", "--parquet_data_dir", default="parquet_files",
                    help="Path to PREPROCESSED parquet data directory for TRAINING")
parser.add_argument("-metadata_file", "--metadata_file", default="QCed_samples_type3.csv",
                    help="Path to the PREPROCESSED QCed_samples_type3.csv metadata file for TRAINING")
parser.add_argument('--no_wandb', action='store_true', default=False,
                    help="Disable wandb logging")

args = parser.parse_args()
# LOG_WANDB = not args.no_wandb
LOG_WANDB = False

# Load config from JSON file
with open(args.config_file, 'r') as f:
    config_from_file = json.load(f)

if config_from_file.get("do_train", True): # Only init wandb if training
    try:
        if LOG_WANDB:
            import wandb
            wandb.login()
            run = wandb.init(
                project="MethyGPT",
                name="pretrain-example-data",
                config=config_from_file,
                save_code=True,
            )
            run.log_code(".")
            print(f"Wandb initialized")
        else:
            print("Wandb logging disabled")
            wandb = None
    except ImportError:
        print("Wandb not found, skipping wandb initialization.")
        wandb = None
    except Exception as e:
        print(f"Error initializing wanb: {e}")
        wandb = None
else:
    wandb = None

# Creates a vocab mapping CpG probe IDs to unique integer indices for model input
probe_id_path = Path(args.probe_id_path)
pad_token = config_from_file.get("pad_token", "<pad>")
special_tokens = config_from_file.get("special_tokens", ["<pad>", "<cls>", "<eoc>"])
methyl_vocab = MethylVocab(
    probe_id_path, pad_token, special_tokens, save_dir=None
)  # Pass None for save_dir initially

config = dict(
    # Important thing to control
    seed=config_from_file.get("seed", 42),
    parquet_dir=Path(args.parquet_data_dir),                # USE PREPROCESSED PARQUET DIR
    probe_id_dir=probe_id_path,                             # This is for vocab, uses the original probe_id_path arg
    data_dir=Path(args.metadata_file),                      # USE PREPROCESSED METADATA FILE
    valid_ratio=config_from_file.get("valid_ratio", 0.1),
    max_fi=config_from_file.get("max_fi", 500000),          # To use full dataset, Just set >500000
    do_train=config_from_file.get("do_train", True),
    pretrained_file=config_from_file.get("pretrained_file", None),  # None for pretraining from scratch
    mask_ratio=config_from_file.get("mask_ratio", 0.3),
    GEPC=config_from_file.get("GEPC", True),                # Masked value prediction for cell embedding
    dab_weight=config_from_file.get("dab_weight", 1.0),

    # Model and training
    epochs=config_from_file.get("epochs", 100),
    ecs_thres=config_from_file.get("ecs_thres", 0.0),  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    lr=config_from_file.get("lr", 1e-3),
    batch_size=config_from_file.get("batch_size", 32),  #4,
    layer_size=config_from_file.get("layer_size", 64),  #16,
    nlayers=config_from_file.get("nlayers", 6),         #4,
    nhead=config_from_file.get("nhead", 4),
    dropout=config_from_file.get("dropout", 0.1),
    schedule_ratio=config_from_file.get("schedule_ratio", 0.9),  # ratio of epochs for learning rate schedule
    save_epoch_interval=config_from_file.get("save_epoch_interval", 10),
    log_batch_interval=config_from_file.get("log_batch_interval", 1000),
    fast_transformer=config_from_file.get("fast_transformer", True) and flash_attn_available, # Ensure flash_attn is available
    pre_norm=config_from_file.get("pre_norm", False),
    amp=config_from_file.get("amp", True) and amp_available, # Ensure amp is available

    # Additional tokens and values
    pad_token=pad_token,
    special_tokens=special_tokens,
    mask_value=config_from_file.get("mask_value", -1),
    pad_value=config_from_file.get("pad_value", -2),
    explicit_zero_prob=config_from_file.get("explicit_zero_prob", False),  # Flag for explicit zero probability
    max_seq_len=len(methyl_vocab.CpG_list) + 1,                   # Use length of CpG list from vocab + 1 for model
    per_seq_batch_sample=config_from_file.get("per_seq_batch_sample", False),  # Flag for per-sequence batch sampling
)

# Update the main config with command-line arguments and W&B run ID if available
config["device"] = device
config["savename"] = args.savename
config["metadata_file"] = args.metadata_file
if wandb and wandb.run:
    config["wandb_run_id"] = wandb.run.id
    wandb.config.update(config, allow_val_change=True) # Log the final combined config to W&B

config_hash = make_hash(config)
set_seed(config["seed"])

# Dir for saving logs, config, vocab and checkpoints of this run
# save_dir = Path(f"save/dev_{config['savename']}-{time.strftime('%b%d-%H-%M')}/")
save_dir = Path(f"save/dev_{config['savename']}/")
save_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger("logger", save_dir / "run.log")
add_console_handler(logger)
train_logger = setup_logger("train_logger", save_dir / "train.log")
add_console_handler(train_logger)
test_logger = setup_logger("test_logger", save_dir / "test.log")
add_console_handler(test_logger)

save_config(config, save_dir)
logger.info(f"Config saved to {save_dir}")

if not (save_dir / "vocab.json").exists():
    methyl_vocab.save_dir = save_dir
    methyl_vocab._save_vocab()
    logger.info(f"MethylVocab saved to {save_dir}")
else:
    methyl_vocab.save_dir = save_dir
    logger.info(f"MethylVocab already exists at {save_dir} or will be loaded from there.")

# ------------------------ Data Preparation ------------------------
parquet_dirs = []
if config["parquet_dir"].exists() and config["parquet_dir"].is_dir():
    parquet_dirs = [
        os.path.join(config["parquet_dir"], f) for f in os.listdir(config["parquet_dir"]) if f.endswith(".parquet")
    ]
else:
    logger.error(f"Parquet directory {config['parquet_dir']} does not exist or is not a directory.")
logger.info(f"Number of parquet files found: {len(parquet_dirs)}")
if not parquet_dirs:
    logger.error("No parquet files found. Please check the preprocessing step and parquet_data_dir path.")
    # Depending on desired behavior, could exit here:
    # sys.exit("Exiting due to no parquet files found.")

train_files, valid_files = split_files(parquet_dirs, valid_ratio=config["valid_ratio"])
logger.info(f"Loading data from {len(train_files)} training files and {len(valid_files)} validation files")
# NUM_WORKER = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()-2))
NUM_WORKER = 0  # Set to 0 for debugging
train_dataloader = create_dataloader(train_files, config["batch_size"], num_workers=NUM_WORKER)
valid_dataloader = create_dataloader(valid_files, config["batch_size"], num_workers=NUM_WORKER)

# ------------------------ Model Architecture & Training Setup ------------------------
logger.info(f"Using device: {device}")
model = MethylGPTModel(
    config=config,  # Pass the entire config dict
    vocab=methyl_vocab
)
model.to(device)

if config["pretrained_file"] is not None:
    try:
        model.load_state_dict(torch.load(config["pretrained_file"], map_location=device))
        logger.info(f"Loaded pretrained model from {config['pretrained_file']}")
    except FileNotFoundError:
        logger.error(f"Pretrained model file not found: {config['pretrained_file']}")
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=config["schedule_ratio"]
)  # step_size=1 if called every epoch

scaler = None
if config["amp"] and device.startswith("cuda"):
    if amp_available:
        scaler = GradScaler()
        logger.info("AMP GradScaler initialized.")
    else: # amp was configured True but torch.cuda.amp.GradScaler is not available
        logger.warning("AMP was configured but GradScaler is not available. Disabling AMP for training.")
        config["amp"] = False # Correctly disable AMP if scaler cannot be used.
else: # AMP not configured or device is CPU
    config["amp"] = False

# Load latest checkpoint (with matching config hash and largest epoch) and resume training,
# otherwise start from scratch
ckpt_dir = Path('./checkpoints')
ckpt_dir.mkdir(parents=True, exist_ok=True)
latest_ckpt_dir = None
if ckpt_dir.exists():
    ckpt_files = list(ckpt_dir.glob(f'checkpoint_{config_hash}_*.pth'))
    if ckpt_files:
        ckpts_with_epochs = []
        for path in ckpt_files:
            try:
                epoch_num = int(path.stem.split('_')[-1])
                ckpts_with_epochs.append((path, epoch_num))
            except ValueError:
                logger.warning(f"Could not parse epoch from checkpoint filename: {path.name}")
        if ckpts_with_epochs:
            latest_ckpt_dir, latest_epoch = max(ckpts_with_epochs, key=lambda x: x[1])
            latest_ckpt_dir = str(latest_ckpt_dir)  # Convert Path to str for torch.load

start_epoch = 1
if latest_ckpt_dir is not None:
    try:
        ckpt = torch.load(latest_ckpt_dir, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']} from {latest_ckpt_dir}")
    except Exception as e:
        logger.error(f"Error loading checkpoint {latest_ckpt_dir}: {e}. Starting from scratch.")
        start_epoch = 1
else:
    logger.info("No checkpoint found. Starting from scratch.")

# A specific checkpoint dir for this run
current_run_ckpt_dir = save_dir / "checkpoints"
current_run_ckpt_dir.mkdir(parents=True, exist_ok=True)

# ------------------------ Training Loop ------------------------
best_val_loss = float("inf")
best_model_state = None
best_model_epoch = 0

logger.info(f"Starting training from epoch {start_epoch} to {config['epochs']}")
for epoch in range(start_epoch, config["epochs"] + 1):
    logger.info(f"--- Epoch {epoch}/{config['epochs']} ---")

    # --------------- Training Phase ---------------
    model.train()
    total_train_loss = 0
    total_train_mse = 0
    total_train_gepc = 0
    processed_train_samples = 0  # Count samples for average loss

    pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch}", leave=False)
    for i, batch in enumerate(pbar_train):
        optimizer.zero_grad()

        # batch: {'id': list of bs samples ids, 'data': tensor of shape [bs, num_CpG_sites]}
        prepared_batch = model.prepare_data(batch)
        # tensors of shape [bs, num_CpG_sites + 1]
        input_gene_ids = prepared_batch["gene_ids"].to(device)      # <cls> + CpG site ids
        input_values = prepared_batch["values"].to(device)          # masked beta values (only mask non-padded positions)
        target_values = prepared_batch["target_values"].to(device)  # original padded beta values

        #TODO: the special tokens <pad> and <eoc> used in the probe id vocab are not really needed here,
        # as the pad_value in the beta values already indicates padding positions (for missing values or to pad to max length)
        # to create the attention mask.

        # padding attention mask: boolean tensor [bs, num_CpG_sites + 1]
        src_key_padding_mask = target_values.eq(config["pad_value"])

        with torch.cuda.amp.autocast(enabled=(config["amp"] and scaler is not None)):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                MVC=config["GEPC"],
                ECS=config["ecs_thres"] > 0
            )

            # Only compute MSE loss on masked positions
            loss_positions = input_values.eq(config["mask_value"])
            loss_mse = masked_mse_loss(output_dict["mlm_output"], target_values, loss_positions)
            loss = loss_mse

            if config["GEPC"] and "mvc_output" in output_dict and output_dict["mvc_output"] is not None:
                loss_gepc = masked_mse_loss(output_dict["mvc_output"], target_values, loss_positions)
                loss = loss + loss_gepc
            else:
                loss_gepc = torch.tensor(0.0).to(device) # So it can be added to total

        if scaler is not None:
            scaler.scale(loss).backward()
            # Consider gradient clipping here if it was in the original and scaler is used
            # scaler.unscale_(optimizer) # Optional if you need to inspect/clip grads before optimizer.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Example clipping
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Example clipping
            optimizer.step()

        bs = input_gene_ids.size(0)
        total_train_loss += loss.item() * bs
        total_train_mse += loss_mse.item() * bs
        if config["GEPC"]:
            total_train_gepc += loss_gepc.item() * bs
        processed_train_samples += bs

        if i % config["log_batch_interval"] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss_so_far = total_train_loss / processed_train_samples if processed_train_samples > 0 else 0
            pbar_train.set_postfix_str(f"Batch Loss: {loss.item():.4f}, Avg Loss: {avg_loss_so_far:.4f}, LR: {current_lr:.2e}")
            if wandb and wandb.run:
                log_dict = {
                    "train_loss_batch": loss.item(),
                    "train_mse_batch": loss_mse.item(),
                    "learning_rate": current_lr,
                    "epoch_progress": epoch + i / len(train_dataloader)
                }
                if config["GEPC"]:
                    log_dict["train_gepc_batch"] = loss_gepc.item()
                # Global step to align with actual training progress (especially after resuming from checkpoints)
                wandb.log(log_dict, step=epoch * len(train_dataloader) + i)

    avg_train_loss = total_train_loss / processed_train_samples if processed_train_samples > 0 else 0
    avg_train_mse = total_train_mse / processed_train_samples if processed_train_samples > 0 else 0
    avg_train_gepc = total_train_gepc / processed_train_samples if processed_train_samples > 0 else 0

    train_logger.info(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f} | Avg Train MSE: {avg_train_mse:.4f} | Avg Train GEPC: {avg_train_gepc:.4f}")
    if wandb and wandb.run:
        log_dict_epoch = {
            "avg_train_loss_epoch": avg_train_loss,
            "avg_train_mse_epoch": avg_train_mse,
            "epoch": epoch
        }
        if config["GEPC"]:
            log_dict_epoch["avg_train_gepc_epoch"] = avg_train_gepc
        wandb.log(log_dict_epoch)

    # --------------- Evaluation Phase ---------------
    model.eval()
    total_valid_loss = 0
    total_valid_mse = 0
    total_valid_gepc = 0
    processed_valid_samples = 0

    pbar_valid = tqdm(valid_dataloader, desc=f"Validating Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch in pbar_valid:
            prepared_batch_valid = model.prepare_data(batch)

            input_gene_ids_valid = prepared_batch_valid["gene_ids"].to(device)
            input_values_valid = prepared_batch_valid["values"].to(device)
            target_values_valid = prepared_batch_valid["target_values"].to(device)

            src_key_padding_mask_valid = target_values_valid.eq(config["pad_value"])

            with torch.cuda.amp.autocast(enabled=(config["amp"] and scaler is not None)):
                output_dict_valid = model(
                    input_gene_ids_valid,
                    input_values_valid,
                    src_key_padding_mask=src_key_padding_mask_valid,
                    MVC=config["GEPC"],
                    ECS=config["ecs_thres"] > 0
                )

                loss_positions_valid = input_values_valid.eq(config["mask_value"])
                loss_mse_valid = masked_mse_loss(output_dict_valid["mlm_output"], target_values_valid, loss_positions_valid)
                loss_valid = loss_mse_valid

                if config["GEPC"] and "mvc_output" in output_dict_valid and output_dict_valid["mvc_output"] is not None:
                    loss_gepc_valid = masked_mse_loss(output_dict_valid["mvc_output"], target_values_valid, loss_positions_valid)
                    loss_valid = loss_valid + loss_gepc_valid
                else:
                    loss_gepc_valid = torch.tensor(0.0).to(device)

            bs_valid = input_gene_ids_valid.size(0)
            total_valid_loss += loss_valid.item() * bs_valid
            total_valid_mse += loss_mse_valid.item() * bs_valid
            if config["GEPC"]:
                total_valid_gepc += loss_gepc_valid.item() * bs_valid
            processed_valid_samples += bs_valid

    avg_valid_loss = total_valid_loss / processed_valid_samples if processed_valid_samples > 0 else 0
    avg_valid_mse = total_valid_mse / processed_valid_samples if processed_valid_samples > 0 else 0
    avg_valid_gepc = total_valid_gepc / processed_valid_samples if processed_valid_samples > 0 else 0

    test_logger.info(f"Epoch {epoch} | Avg Valid Loss: {avg_valid_loss:.4f} | Avg Valid MSE: {avg_valid_mse:.4f} | Avg Valid GEPC: {avg_valid_gepc:.4f}")
    if wandb and wandb.run:
        log_dict_epoch_val = {
            "avg_valid_loss_epoch": avg_valid_loss,
            "avg_valid_mse_epoch": avg_valid_mse,
            "epoch": epoch
        }
        if config["GEPC"]:
            log_dict_epoch_val["avg_valid_gepc_epoch"] = avg_valid_gepc
        wandb.log(log_dict_epoch_val)

    # Lr sheduled at each epoch end
    lr_scheduler.step()

    # --------------- Checkpointing ---------------
    if epoch % config["save_epoch_interval"] == 0 or epoch == config["epochs"]:
        ckpt_epoch_dir = current_run_ckpt_dir / f"checkpoint_{config_hash}_{epoch}.pth"
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            }, ckpt_epoch_dir)
            logger.info(f"Checkpoint saved at epoch {epoch}: {ckpt_epoch_dir}")
        except Exception as e:
            logger.error(f"Error saving checkpoint at epoch {epoch}: {e}")

    # Early stopping or other criteria could be checked here
    # For now, we just log the best model based on validation loss
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_model_epoch = epoch
        logger.info(f"New best model found at epoch {epoch} with validation loss: {best_val_loss:.4f}")

# ------------------------ Final Evaluation ------------------------
# At the end of training, load the best model state if available
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    logger.info(f"Loaded best model from epoch {best_model_epoch} with validation loss: {best_val_loss:.4f}")
else:
    logger.warning("No best model found (best_model_state is None). Training may not have completed successfully.")

# Final evaluation on the validation set with the best model
# This section might need adjustment if UMAP or other specific evaluations are required as per original.
# For now, it mirrors the validation loop structure for loss calculation.
model.eval()
total_final_valid_loss = 0
total_final_valid_mse = 0
total_final_valid_gepc = 0
processed_final_valid_samples = 0

pbar_final_valid = tqdm(valid_dataloader, desc="Final Validation", leave=False)
with torch.no_grad():
    for batch in pbar_final_valid:
        prepared_batch_final = model.prepare_data(batch)

        input_gene_ids_final = prepared_batch_final["gene_ids"].to(device)
        input_values_final = prepared_batch_final["values"].to(device)
        target_values_final = prepared_batch_final["target_values"].to(device)

        src_key_padding_mask_final = target_values_final.eq(config["pad_value"])

        with torch.cuda.amp.autocast(enabled=(config["amp"] and scaler is not None)):
            output_dict_final = model(
                input_gene_ids_final,
                input_values_final,
                src_key_padding_mask=src_key_padding_mask_final,
                MVC=config["GEPC"],
                ECS=config["ecs_thres"] > 0
            )

            loss_positions_final = input_values_final.eq(config["mask_value"])
            loss_mse_final = masked_mse_loss(output_dict_final["mlm_output"], target_values_final, loss_positions_final)
            loss_final = loss_mse_final

            if config["GEPC"] and "mvc_output" in output_dict_final and output_dict_final["mvc_output"] is not None:
                loss_gepc_final = masked_mse_loss(output_dict_final["mvc_output"], target_values_final, loss_positions_final)
                loss_final = loss_final + loss_gepc_final
            else:
                loss_gepc_final = torch.tensor(0.0).to(device)

        bs_final = input_gene_ids_final.size(0)
        total_final_valid_loss += loss_final.item() * bs_final
        total_final_valid_mse += loss_mse_final.item() * bs_final
        if config["GEPC"]:
            total_final_valid_gepc += loss_gepc_final.item() * bs_final
        processed_final_valid_samples += bs_final

avg_final_valid_loss = total_final_valid_loss / processed_final_valid_samples if processed_final_valid_samples > 0 else 0
avg_final_valid_mse = total_final_valid_mse / processed_final_valid_samples if processed_final_valid_samples > 0 else 0
avg_final_valid_gepc = total_final_valid_gepc / processed_final_valid_samples if processed_final_valid_samples > 0 else 0

logger.info(f"Final Validation | Avg Loss: {avg_final_valid_loss:.4f} | Avg MSE: {avg_final_valid_mse:.4f} | Avg GEPC: {avg_final_valid_gepc:.4f}")

if wandb and wandb.run:
    wandb.finish()