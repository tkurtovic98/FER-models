from .vgg import *
from .resnet import *

_models = {
    'VGG19': VGG('VGG19'),
    'Resnet18': ResNet18()
}

def get_model(model_name: str):
    if _models.get(model_name) is None:
        raise ValueError("Unknown model name")
    return _models[model_name]