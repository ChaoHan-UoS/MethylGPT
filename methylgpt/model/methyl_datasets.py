import os
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path

class CustomDataset(IterableDataset):
    def __init__(self, parquet_chunk_files, input_normalization=False): # Removed pad_value
        """
        Dataset to read from Parquet files where each file contains a chunk of cells.
        Each row in a Parquet file is expected to have an 'id' (cell_id) and 
        a 'data' column (list of methylation values, already ordered by preprocessing).
        """
        self.parquet_chunk_files = parquet_chunk_files
        # self.pad_value = pad_value # Removed
        self.input_normalization = input_normalization

        if self.input_normalization:
            raise NotImplementedError("Input normalization has not been implemented")

    def __iter__(self):
        for file_path_str in self.parquet_chunk_files:
            file_path = Path(file_path_str)
            try:
                df_chunk = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Error reading Parquet chunk file {file_path}: {e}")
                continue

            if 'id' not in df_chunk.columns or 'data' not in df_chunk.columns:
                print(f"Warning: 'id' or 'data' column not found in Parquet chunk {file_path}. Skipping this file.")
                continue
            
            for index, row in df_chunk.iterrows():
                cell_id = str(row['id'])
                methylation_values_list = row['data']
                
                if not isinstance(methylation_values_list, (list, np.ndarray)):
                    print(f"Warning: 'data' for cell_id {cell_id} in chunk {file_path} is not a list or ndarray. Type: {type(methylation_values_list)}. Skipping cell.")
                    continue
                
                try:
                    # Ensure data_array is a 1D numpy array of floats
                    data_array = np.array(methylation_values_list, dtype=float)
                    if data_array.ndim != 1:
                         print(f"Warning: 'data' for cell_id {cell_id} is not 1D. Shape: {data_array.shape}. Skipping cell.")
                         continue
                except ValueError as ve:
                    print(f"Warning: Could not convert 'data' to numeric array for cell_id {cell_id} in chunk {file_path}. Error: {ve}. Skipping cell.")
                    continue
                
                # Yields raw data; tokenization/masking will be done in the training script
                yield {"id": cell_id, "data": data_array}


def split_files(files, valid_ratio):
    num_files = len(files)
    # np.random.shuffle(files) # Optional: shuffle before splitting
    split_index = int(np.floor(num_files * (1 - valid_ratio)))
    return files[:split_index], files[split_index:]


def create_dataloader(parquet_chunk_files, batch_size, num_workers=None, max_workers=8): # Removed pad_value
    # methyl_vocab and config removed from arguments
    dataset = CustomDataset(parquet_chunk_files) # Removed pad_value=pad_value
    
    effective_num_files = len(parquet_chunk_files)
    if effective_num_files == 0:
        print("Warning: No Parquet chunk files provided to create_dataloader. DataLoader will be empty.")
        num_workers = 0

    if num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
            num_workers = min(num_cpus, max_workers, effective_num_files if effective_num_files > 0 else 1)
            num_workers = max(1 if effective_num_files > 0 else 0, num_workers)
        except AttributeError: 
            num_cpus = os.cpu_count()
            num_workers = min(num_cpus if num_cpus is not None else 1, max_workers, effective_num_files if effective_num_files > 0 else 1)
            num_workers = max(1 if effective_num_files > 0 else 0, num_workers)
        except Exception:
            num_workers = min(4, max_workers, effective_num_files if effective_num_files > 0 else 1)
            num_workers = max(1 if effective_num_files > 0 else 0, num_workers)
    
    if effective_num_files == 0: # Ensure num_workers is 0 if no files
        num_workers = 0

    dataloader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        dataloader_args["prefetch_factor"] = 2 
        # dataloader_args["persistent_workers"] = True # Consider for IterableDataset if appropriate
    
    return DataLoader(
        dataset, 
        **dataloader_args
    )
