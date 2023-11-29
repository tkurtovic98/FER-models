from __future__ import print_function

import os
import argparse
import torch
from torch import nn, optim

from dataset_loaders import get_dataset_loader

from models import get_model

from plot_progress import plot_progress
from training_state import TrainingState
from learning import LearningRateDecay

from learning import train, run_validation

from checkpoint import load_checkpoint, set_checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
    parser.add_argument('--model', type=str, default='VGG19',
                        help='Model architecture.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use.')
    parser.add_argument('--bs', default=128, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint.')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of epochs to run')                
    parser.add_argument('--root', required=True, type=str, help='Root dir where models are stored and loaded from.')
    parser.add_argument('-lds', type=int, default=70, help='Epoch at which to start learning rate decay.')
    parser.add_argument('-lde', type=int, default=5, help='Decrease the learning rate every number of epochs.')
    parser.add_argument('-ldr', type=float, default=0.9, help='Rate of learning rate decay.')
    
    parser.add_argument('--fold', type=int, default=1, help="Used for k-fold algorithm")
    
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()

    root_path = opt.root 
    name = f'{opt.dataset}_{opt.model}'
    path = os.path.join(root_path, name)

    set_checkpoint_path(path)

    net = get_model(opt.model)

    dataset_loader = get_dataset_loader(opt.dataset)

    if opt.dataset == "CK":
        train_set_loader, validation_set_loader = dataset_loader(opt.bs, opt.fold) #Quick fix
    else:
        train_set_loader, validation_set_loader, test_set_loader = dataset_loader(opt.bs)

    state = TrainingState()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print(f'Cuda device active: {torch.cuda.get_device_name(0)}')
    else:
        print("Cuda device not active...")

    start_epoch = 0

    if opt.resume:
        checkpoint = load_checkpoint('PrivateTest_model.t7')

        net.load_state_dict(checkpoint.net)

        state.best_test_acc = checkpoint.best_val_acc
        state.best_test_acc_epoch = checkpoint.best_val_epoch

        state.best_validation_acc = checkpoint.best_test_acc
        state.best_validation_acc_epoch = checkpoint.best_val_epoch

        start_epoch = checkpoint.best_val_epoch + 1
    else:
        print('==> Building model..')

    if use_cuda:
        net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    learning_rate = opt.lr
    learning_rate_decay = LearningRateDecay(start = opt.lds, every=opt.lde, rate=opt.ldr)

    for epoch in range(start_epoch, opt.epoch):
        state.set_epoch(epoch)
        state = train(epoch, state, learning_rate_decay, net, train_set_loader, learning_rate, optimizer, loss_fn)
        state = run_validation(epoch, state, net, validation_set_loader, learning_rate, optimizer, loss_fn)
        plot_progress(state, name, img_path = root_path)

    best_validation_acc, best_validation_epoch = state.get_best_validation_acc()

    print(
        f"Best Private Test Accuracy: {best_validation_acc} at Epoch {best_validation_epoch}")
