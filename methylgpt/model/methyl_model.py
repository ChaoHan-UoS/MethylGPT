import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent / "modules" / "scGPT"))
# current_directory = Path(__file__).parent.absolute()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from scgpt.model.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import torch.nn.functional as F
from sklearn import preprocessing
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import torch
# import lightning as pl
from sklearn.linear_model import ElasticNet
import math
import pickle
from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union


class MethylGPTModel(TransformerModel):
    def __init__(self, config, vocab):
        super().__init__(
            len(vocab),            # 49159
            config["layer_size"],
            config["nhead"],
            config["layer_size"],
            config["nlayers"],
            vocab=vocab,
            dropout=config["dropout"],
            pad_token=vocab.pad_token,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=None,
            domain_spec_batchnorm=False,
            n_input_bins=None,
            ecs_threshold=None,
            explicit_zero_prob=False,
            use_fast_transformer=config["fast_transformer"],
            pre_norm=config["pre_norm"])
        self.vocab = vocab
        self.config= config
        self.validation_step_outputs = []
        
    def get_cell_embeddings(self, gene_ids, values):
        # with torch.no_grad():
        #     model.eval()
        src_key_padding_mask = gene_ids.eq(self.vocab[self.vocab.pad_token])
        output_dict = self(
            gene_ids,
            values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
        )
        cell_embeddings = output_dict["cell_emb"]
            
        return cell_embeddings
    
    @classmethod
    def from_pretrained(self, config, vocab):
        if config["load_model"]:
            try:
                self.load_state_dict(torch.load(config["pretrained_file"], map_location="cpu"))
                print(f'Loading all model params from {config["pretrained_file"]}')
            except:
                # only load params that are in the model and match the size
                model_dict = self.state_dict()
                pretrained_dict = torch.load(config["pretrained_file"], map_location="cpu")
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict)

    def prepare_data(self, batch):
        """
        batch: {'id': list of bs sample ids, 'data': Tensor of shape [bs, num_CpG_sites]}
        Prepares the input data by tokenizing, padding, and applying random masking.
        """
        # Retrieve config values
        max_seq_len = self.config['max_seq_len']
        mask_ratio = self.config['mask_ratio']
        mask_value = self.config['mask_value']
        pad_token = self.config['pad_token']
        pad_value = self.config['pad_value']
        append_cls = self.config.get("append_cls", True)
        include_zero_gene = self.config.get("include_zero_gene", True)  # From your snippet; scGPT often False

        # Replace NaN (missing from reference probe IDs) with pad_value
        methyl_data_tensor = batch["data"]
        methyl_data_tensor = methyl_data_tensor.float()
        methyl_data_tensor = torch.nan_to_num(methyl_data_tensor, nan=float(pad_value))

        # Ensure tensor is on CPU and convert it to array for the tokenizer
        methyl_data_numpy = methyl_data_tensor.cpu().numpy()

        if not hasattr(self.vocab, 'CpG_ids'):
            raise AttributeError("MethylVocab instance does not have 'CpG_ids' attribute. Please verify attribute name for the list of all CpG sites.")

        # A dict of CpG site ids and their beta value tensors of shape (bs, n_CpGs + 1):
        # {'genes': tensor([[1, 3, 4, ... 49158], ... ],
        #  'values': tensor([[ 0.0000, -2.0000, 0.3023, ... -2.0000], ... ])}
        tokenized_data = tokenize_and_pad_batch(
            methyl_data_numpy,          # 2D array
            self.vocab.CpG_ids,         # all CpG IDs from the vocab: array([3, 4, 5, ... 49158])
            max_len=max_seq_len,
            vocab=self.vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=append_cls, 
            include_zero_gene=include_zero_gene,
        )

        # Value-level random masking of the beta values
        masked_input_values = random_mask_value(
            tokenized_data["values"], 
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value, 
        )

        return {
            "gene_ids": tokenized_data["genes"],
            "values": masked_input_values,
            "target_values": tokenized_data["values"],
        }
