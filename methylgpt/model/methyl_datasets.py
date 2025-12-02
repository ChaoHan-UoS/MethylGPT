import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch 
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class CustomDataset(IterableDataset):
    def __init__(
            self, parquet_chunk_files, input_normalization=False, ddp_world_size=1, ddp_rank=0, seed=42
    ):
        """
        Dataset to read from Parquet files where each file contains a chunk of cells.
        Each row in a Parquet file is expected to have an 'id' (cell_id) and 
        a 'data' column (list of methylation values, already ordered by preprocessing).
        """
        self.parquet_chunk_files = parquet_chunk_files
        self.input_normalization = input_normalization
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.seed = seed
        self.epoch = 0  # will be set from training loop

        if self.input_normalization:
            raise NotImplementedError("Input normalization has not been implemented")

        # Cache counts
        self._total_length = None      # all ranks
        self._local_length = None      # this rank only

    def total_len(self):
        """Total number of valid samples across ALL ranks (all files)."""
        if self._total_length is not None:
            return self._total_length

        count = 0
        for file_path_str in self.parquet_chunk_files:
            file_path = Path(file_path_str)
            try:
                df_chunk = pd.read_parquet(file_path)
            except Exception:
                continue
            if 'id' not in df_chunk.columns or 'data' not in df_chunk.columns:
                continue

            for _, row in df_chunk.iterrows():
                methylation_values_list = row['data']
                if not isinstance(methylation_values_list, (list, np.ndarray)):
                    continue
                try:
                    data_array = np.array(methylation_values_list, dtype=float)
                    if data_array.ndim != 1:
                        continue
                except ValueError:
                    continue
                count += 1

        self._total_length = count
        return self._total_length

    def _rank_file_indices(self):
        """Files assigned to this DDP rank (simple modulo sharding)."""
        n_files = len(self.parquet_chunk_files)
        return [i for i in range(n_files) if i % self.ddp_world_size == self.ddp_rank]

    def local_len(self):
        """Number of valid samples for THIS rank (before DataLoader workers)."""
        if self._local_length is not None:
            return self._local_length

        count = 0
        for idx in self._rank_file_indices():
            file_path_str = self.parquet_chunk_files[idx]
            file_path = Path(file_path_str)
            try:
                df_chunk = pd.read_parquet(file_path)
            except Exception:
                continue
            if 'id' not in df_chunk.columns or 'data' not in df_chunk.columns:
                continue

            for _, row in df_chunk.iterrows():
                methylation_values_list = row['data']
                if not isinstance(methylation_values_list, (list, np.ndarray)):
                    continue
                try:
                    data_array = np.array(methylation_values_list, dtype=float)
                    if data_array.ndim != 1:
                        continue
                except ValueError:
                    continue
                count += 1

        self._local_length = count
        return self._local_length

    def _iter_files(self, file_indices):
        for idx in file_indices:
            file_path_str = self.parquet_chunk_files[idx]
            file_path = Path(file_path_str)
            try:
                df_chunk = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Error reading Parquet chunk file {file_path}: {e}")
                continue

            if 'id' not in df_chunk.columns or 'data' not in df_chunk.columns:
                print(f"Warning: 'id' or 'data' column not found in Parquet chunk {file_path}. Skipping this file.")
                continue

            for _, row in df_chunk.iterrows():
                cell_id = str(row['id'])
                methylation_values_list = row['data']

                if not isinstance(methylation_values_list, (list, np.ndarray)):
                    print(f"Warning: 'data' for cell_id {cell_id} in chunk {file_path} is not a list or ndarray. "
                          f"Type: {type(methylation_values_list)}. Skipping cell.")
                    continue

                try:
                    # Ensure data_array is a 1D numpy array of floats
                    data_array = np.array(methylation_values_list, dtype=float)
                    if data_array.ndim != 1:
                        print(f"Warning: 'data' for cell_id {cell_id} is not 1D. Shape: {data_array.shape}. Skipping cell.")
                        continue
                except ValueError as ve:
                    print(f"Warning: Could not convert 'data' to numeric array for cell_id {cell_id} in chunk {file_path}. "
                          f"Error: {ve}. Skipping cell.")
                    continue

                # Yields raw data; tokenization/masking will be done in the training script
                yield {"id": cell_id, "data": data_array}

    def __iter__(self):
        """
        Shard files across:
          1) DDP ranks (by modulo on file index)
          2) DataLoader workers within each rank
        and shuffle the file order per rank per epoch.
        """
        worker_info = get_worker_info()
        if worker_info is None:
            num_workers = 1  # the main process (no worker process)
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        rank_files = self._rank_file_indices()
        n_rank_files = len(rank_files)
        if n_rank_files == 0:
            return iter(())  # empty iterator

        # Shuffle file order for this rank, deterministically per epoch
        g = torch.Generator()
        g.manual_seed(self.seed + 12345 * self.epoch + 31 * self.ddp_rank)
        perm = torch.randperm(n_rank_files, generator=g).tolist()
        rank_files = [rank_files[i] for i in perm]

        # Split these rank-local files across workers on this rank
        files_per_worker = int(math.ceil(n_rank_files / num_workers))
        start = worker_id * files_per_worker
        end = min(start + files_per_worker, n_rank_files)
        file_indices = rank_files[start:end]

        return self._iter_files(file_indices)


def split_files(files, valid_ratio):
    num_files = len(files)
    # np.random.shuffle(files) # Optional: shuffle before splitting
    split_index = int(np.floor(num_files * (1 - valid_ratio)))
    return files[:split_index], files[split_index:]


def create_dataloader(
        parquet_chunk_files, batch_size, num_workers=None, max_workers=12, ddp_world_size=1, ddp_rank=0, seed=42
):
    dataset = CustomDataset(
        parquet_chunk_files, ddp_world_size=ddp_world_size, ddp_rank=ddp_rank, seed=seed
    )

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

    dataloader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        dataloader_args["prefetch_factor"] = 2 
        # dataloader_args["persistent_workers"] = True # Consider for IterableDataset if appropriate
    
    return DataLoader(dataset, **dataloader_args)
