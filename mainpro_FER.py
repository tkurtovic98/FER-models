'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function
from plot_progress import plot_progress
from training_state import TrainingState

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import utils
from torch.autograd import Variable
from models import *
from fer import prepare_dataset

from learning import LearningRateDecay

from learning import train, run_testing, run_validation

from checkpoint import load_checkpoint, set_checkpoint_path

total_epoch = 250

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
    parser.add_argument('--model', type=str, default='VGG19',
                        help='CNN architecture')
    parser.add_argument('--dataset', type=str,
                        default='FER2013', help='CNN architecture')
    parser.add_argument('--bs', default=128, type=int, help='learning rate')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_args()

    path = os.path.join(opt.dataset + '_' + opt.model)

    set_checkpoint_path(path)

    net = get_model(opt.model)

    state = TrainingState()

    use_cuda = torch.cuda.is_available()

    if opt.resume:
        # Load checkpoint.
        checkpoint = load_checkpoint('PrivateTest_model.t7')

        net.load_state_dict(checkpoint.net)

        state.best_PrivateTest_acc = checkpoint.best_val_acc
        state.best_PrivateTest_acc_epoch = checkpoint.best_val_epoch

        state.best_PublicTest_acc = checkpoint.best_test_acc
        state.best_PublicTest_acc_epoch = checkpoint.best_val_epoch

        start_epoch = checkpoint.best_val_epoch + 1
    else:
        print('==> Building model..')
        start_epoch = 0

    if use_cuda:
        net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    learning_rate = opt.lr
    learning_rate_decay = LearningRateDecay(start = 80, every=5, rate=0.9)

    train_set_loader, test_set_loader, validation_set_loader = prepare_dataset(opt.bs)

    for epoch in range(start_epoch, total_epoch):
        state.set_epoch(epoch)
        state = train(epoch, state, learning_rate_decay, use_cuda, net, train_set_loader, learning_rate, optimizer, loss_fn)
        state = run_testing(epoch, state, net, test_set_loader, learning_rate, optimizer, loss_fn)
        state = run_validation(epoch, state, net, validation_set_loader, learning_rate, optimizer, loss_fn)
        plot_progress(state)

    best_public_test_acc, best_public_test_epoch = state.get_best_PublicTest_acc()
    best_private_test_acc, best_private_test_epoch = state.get_best_PrivateTest_acc()

    print(
        f"Best Public Test Accuracy: {best_public_test_acc} at Epoch {best_public_test_epoch}")
    print(
        f"Best Private Test Accuracy: {best_private_test_acc} at Epoch {best_private_test_epoch}")
