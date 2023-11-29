from .CK import prepare_dataset as prepare_ck 
from .FER13 import prepare_dataset as prepare_fer
from .SFEW import prepare_dataset as prepare_sfew
from torch.utils.data import DataLoader

_loader_dict = {
    "CK": prepare_ck,
    "FER2013": prepare_fer,
    "SFEW": prepare_sfew
}

def supported_datasets(): 
    return _loader_dict.keys()

def get_dataset_loader(name: str) :
    assert _loader_dict.get(name) is not None, f'Dataset {name} not supported! [{",".join(supported_datasets())}]'
    return _loader_dict[name]

