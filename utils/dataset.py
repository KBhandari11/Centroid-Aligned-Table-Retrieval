
import numpy as np
import random

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Dataset: GIANT multi-dataset corpus, uses ALL available reps per table
# -----------------------------
class CacheMultiViewDatasetGiant(Dataset):
    """
    Each item is (dataset, table_id) and returns all cached views across reps for that dataset/table.
    """
    def __init__(
        self,
        dataset_to_rep_to_emb: Dict[str, Dict[str, np.memmap]],
        dataset_to_rep_to_row: Dict[str, Dict[str, Dict[str, int]]],
        samples: List[Tuple[str, str]],          # (dataset, tid)
        rep_names: List[str],
        max_views: int = 6, 
        seed: int = 0,
    ):
        self.dataset_to_rep_to_emb = dataset_to_rep_to_emb
        self.dataset_to_rep_to_row = dataset_to_rep_to_row
        self.samples = samples
        self.rep_names = rep_names
        self.max_views = max_views
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def _available_reps(self, dataset: str, tid: str) -> List[str]:
        rep_to_row = self.dataset_to_rep_to_row[dataset]
        rep_to_emb = self.dataset_to_rep_to_emb[dataset]
        reps = [r for r in self.rep_names if (r in rep_to_emb and tid in rep_to_row.get(r, {}))]
        return reps

    def __getitem__(self, idx: int):
        dataset, tid = self.samples[idx]
        reps = self._available_reps(dataset, tid)

        if len(reps) == 0:
            raise RuntimeError(f"No representations available for dataset={dataset} table_id={tid}")
        if len(reps) == 1:
            reps = reps + reps  # duplicate to keep at least 2 views

        if len(reps) > self.max_views:
            reps = self.rng.sample(reps, self.max_views)

        rep_to_emb = self.dataset_to_rep_to_emb[dataset]
        rep_to_row = self.dataset_to_rep_to_row[dataset]

        embs = []
        for r in reps:
            row = rep_to_row[r][tid]
            e = np.asarray(rep_to_emb[r][row], dtype=np.float32)
            embs.append(torch.tensor(e, dtype=torch.float32))

        embs = torch.stack(embs, dim=0)  # (num_views_i, D)
        # IMPORTANT: include dataset in the key to avoid collisions across datasets
        key = f"{dataset}::{tid}"
        return key, embs


def collate_fn(batch):
    """
    batch: [(key, embs[num_views_i,D]), ...] where key is unique string per (dataset, tid)
    Returns:
      table_ids_views: (total_views,) int ids for grouping
      e_views: (total_views, D)
    """
    table_ids_views = []
    e_views = []
    for key, embs in batch:
        m = embs.shape[0]
        hid = (hash(key) % (2**31 - 1))
        table_ids_views.extend([hid] * m)
        e_views.append(embs)
    return torch.tensor(table_ids_views, dtype=torch.long), torch.cat(e_views, dim=0)