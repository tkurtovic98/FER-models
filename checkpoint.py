import torch
from dataclasses import dataclass, asdict
from typing import Union
import os

@dataclass
class Checkpoint():
    net: dict
    best_test_acc: float
    best_test_epoch: float
    best_val_acc: Union[float, None]
    best_val_epoch: Union[float, None]


def load_checkpoint(name:str) -> Checkpoint:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkpoint_path, name))

    return Checkpoint(**checkpoint)


def save_checkpoint(name: str, checkpoint: Checkpoint):
    assert checkpoint_path != '', 'Error: checkpoint path not set!'
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(asdict(checkpoint), os.path.join(checkpoint_path, name))    

checkpoint_path = ''

def set_checkpoint_path(path: str):
    global checkpoint_path
    checkpoint_path = path

