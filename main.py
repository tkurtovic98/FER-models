from __future__ import print_function
import numpy as np

import os
import argparse
import torch
from torch import nn, optim
from torchinfo import summary
from torchvision import models as torchmodels

from dataset_loaders import get_dataset_loader

from models import get_model

from plot_progress import plot_progress
from training_state import TrainingState

from learning import train, run_validation

from checkpoint import load_checkpoint, set_checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch FER CNN Training')
    parser.add_argument('--model', type=str, default='VGG19',
                        help='Model architecture.')
    parser.add_argument('--dataset', type=str,
                        required=True, help='Dataset to use.')
    parser.add_argument('--outname', type=str, required=False, help='Custom output name to use.')
    parser.add_argument('--bs', default=128, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate.')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint.')
    parser.add_argument('--epoch', '-e', type=int,
                        default=100, help='Number of epochs to run')
    parser.add_argument('--root', required=True, type=str,
                        help='Root dir where models are stored and loaded from.')
    parser.add_argument('-lds', type=int, default=70,
                        help='Epoch at which to start learning rate decay.')
    parser.add_argument('-lde', type=int, default=5,
                        help='Decrease the learning rate every number of epochs.')
    parser.add_argument('-ldr', type=float, default=0.9,
                        help='Rate of learning rate decay.')

    parser.add_argument('-v', '--version', type=str,
                        default='1.0', help='Model version')

    parser.add_argument('--split', type=str,
                        default='Validation', help='Dataset split to use.')
    parser.add_argument('--pretrained', '-p', action='store_true', default=False,
                        help='Indicates whether to use pretrained model or not.')

    parser.add_argument('--fold', type=int, default=1,
                        help="Used for k-fold algorithm")

    return parser.parse_args()


# Extract this to config file
label_to_weight: dict[int, float] = {
    0: 7215/3995,
    1: 7215/436,
    2: 7215/4097,
    3: 1,
    4: 7215/4830,
    5: 7215/3171,
    6: 7215/4965
}

label_to_weight: dict[int, float] = {
    0: 193/171,
    1: 193/142,
    2: 1,
    3: 193/83,
    4: 193/154,
    5: 193/62,
    6: 193/91
}

import datetime
import json

def save_config(path: str, name: str,  batch_size: int, decay_start: int, decay_interval: int, decay_factor: int):
    save_dict = {
        "meta": {
            "name": name,
            "start_time": datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        },
        "config": {
            "batch_size": batch_size,
            "decay": {
                "start": decay_start,
                "interval": decay_interval,
                "factor": decay_factor
            }
        }
    }

    os.makedirs(path, exist_ok=True)

    with open(f'{path}/train_config.json', 'w', encoding='utf8') as file:
        json.dump(save_dict, file)



if __name__ == "__main__":
    opt = parse_args()

    root_path = opt.root
    name = f'{opt.dataset if opt.outname is None else opt.outname}_{opt.model}'
    path = os.path.join(root_path, name, opt.version)

    set_checkpoint_path(path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    save_config(path=path, name=name, batch_size=opt.bs, decay_start=opt.lds, decay_interval=opt.lde, decay_factor=opt.ldr)

    if use_cuda:
        print(f'Cuda device active: {torch.cuda.get_device_name(0)}')
    else:
        print("Cuda device not active...")

    net = nn.Module()

    if opt.pretrained:
        print(f'==> Using pretrained model: {opt.model}')

        if opt.model == "VGG19":
            net = torchmodels.vgg19(weights='DEFAULT')

            net.classifier = torch.nn.Linear(25088, 7).to(device)

            for x in net.features.parameters():
                x.requires_grad = False
        elif opt.model == "AlexNet":
            net = torchmodels.alexnet(weights='DEFAULT')
            
            last_cls_in_features = (net.classifier[-1]).in_features
            
            new_last_cls = nn.Linear(last_cls_in_features, 7)
            
            net.classifier[-1] = new_last_cls
        else:
            net = torchmodels.resnet18(weights='DEFAULT')

            num_features = net.fc.in_features
            net.fc = torch.nn.Linear(num_features, 7).to(device)

            net.conv1.requires_grad = False
            net.bn1.requires_grad = False
            net.layer1.requires_grad = False
            net.layer2.requires_grad = False
            net.layer3.requires_grad = False
            net.layer4.requires_grad = False
    else:
        net = get_model(opt.model)
        
    print(net)

    dataset_loader = get_dataset_loader(opt.dataset)

    if opt.dataset == "CK":
        train_set_loader, validation_set_loader = dataset_loader(
            opt.bs, opt.fold)  # Quick fix
    else:
        train_set_loader, validation_set_loader, test_set_loader = dataset_loader(
            opt.bs)

    state = TrainingState()

    start_epoch = 0

    if opt.resume:
        checkpoint = load_checkpoint('Validation_model.t7')

        net.load_state_dict(checkpoint.net)

        # state.best_test_acc = checkpoint.best_val_acc
        # state.best_test_acc_epoch = checkpoint.best_val_epoch

        # state.best_validation_acc = checkpoint.best_test_acc
        # state.best_validation_acc_epoch = checkpoint.best_val_epoch

        # start_epoch = checkpoint.best_val_epoch + 1
    else:
        print('==> Building model..')

    if use_cuda:
        net.cuda()

    # weights = torch.from_numpy(np.array(list(label_to_weight.values()), dtype=np.float32)).to(device)
    # loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    loss_fn = nn.CrossEntropyLoss()    

    learning_rate = opt.lr
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lde, gamma=opt.ldr)

    for epoch in range(start_epoch, opt.epoch):
        state.set_epoch(epoch)
        state = train(epoch, state, net,
                      train_set_loader, learning_rate, optimizer, loss_fn)
        state = run_validation(
            epoch, state, net, validation_set_loader, learning_rate, optimizer, loss_fn)
        if epoch > opt.lds:
            scheduler.step()
        plot_progress(state, name, img_path=path)

    best_validation_acc, best_validation_epoch = state.get_best_validation_acc()

    print(
        f"Best Private Test Accuracy: {best_validation_acc} at Epoch {best_validation_epoch}")
