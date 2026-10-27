"""Dataset helpers for CHARLS person-wave token arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def stack_column(frame: pd.DataFrame, column: str, dtype: np.dtype) -> np.ndarray:
    return np.stack(frame[column].map(lambda value: np.asarray(value, dtype=dtype)).to_numpy())


class PersonWaveMLMDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.raw_ids = frame["ID"].to_numpy(dtype=object)
        id_codes, _ = pd.factorize(frame["ID"].astype(str), sort=True)
        self.ids = id_codes.astype(np.int64)
        self.waves = frame["wave"].to_numpy(dtype=np.int16)
        self.wave_token_ids = frame["wave_token_id"].to_numpy(dtype=np.int16)
        self.value_ids = stack_column(frame, "value_token_ids", np.int16)
        self.trait_ids = stack_column(frame, "trait_token_ids", np.int16)
        self.module_ids = stack_column(frame, "module_token_ids", np.int16)
        self.maskable = stack_column(frame, "value_is_masked_target", np.bool_)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "ID": self.ids[index],
            "raw_ID": self.raw_ids[index],
            "wave": self.waves[index],
            "wave_token_id": self.wave_token_ids[index],
            "value_ids": self.value_ids[index],
            "trait_ids": self.trait_ids[index],
            "module_ids": self.module_ids[index],
            "maskable": self.maskable[index],
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ID": torch.as_tensor([item["ID"] for item in items], dtype=torch.long),
        "raw_ID": [item["raw_ID"] for item in items],
        "wave": torch.as_tensor([item["wave"] for item in items], dtype=torch.long),
        "wave_token_id": torch.as_tensor(
            [item["wave_token_id"] for item in items], dtype=torch.long
        ),
        "value_ids": torch.as_tensor(
            np.stack([item["value_ids"] for item in items]), dtype=torch.long
        ),
        "trait_ids": torch.as_tensor(
            np.stack([item["trait_ids"] for item in items]), dtype=torch.long
        ),
        "module_ids": torch.as_tensor(
            np.stack([item["module_ids"] for item in items]), dtype=torch.long
        ),
        "maskable": torch.as_tensor(
            np.stack([item["maskable"] for item in items]), dtype=torch.bool
        ),
    }
