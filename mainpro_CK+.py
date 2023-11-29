'''Train CK+ with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import os
import argparse
import torch
from torch import nn, optim
from CK import prepare_dataset
from models import get_model
from training_state import TrainingState
from plot_progress import plot_progress

from checkpoint import set_checkpoint_path, load_checkpoint

from learning import LearningRateDecay, train, run_testing

use_cuda = torch.cuda.is_available()

TOTAL_EPOCH = 60

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
    parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
    parser.add_argument('--dataset', type=str, default='CK+', help='dataset')
    parser.add_argument('--fold', default=1, type=int, help='k fold number')
    parser.add_argument('--bs', default=128, type=int, help='batch_size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()

    if use_cuda:
        print(f'Cuda device active: {torch.cuda.get_device_name(0)}')
    else:
        print("Cuda device not active...")
    
    google_drive_path = '/content/drive/MyDrive/FER_Doktorski/FER-models'

    name = opt.dataset + '_' + opt.model

    path = os.path.join(google_drive_path, name, str(opt.fold))

    set_checkpoint_path(path)

    net = get_model(opt.model)

    state = TrainingState()

    if opt.resume:
        checkpoint = load_checkpoint('PrivateTest_model.t7')

        net.load_state_dict(checkpoint.net)

        state.best_PublicTest_acc = checkpoint.best_test_acc
        state.best_PublicTest_acc_epoch = checkpoint.best_test_epoch

        start_epoch = checkpoint.best_test_epoch + 1
    else:
        print('==> Building model..')
        start_epoch = 0

    if use_cuda:
        net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    learning_rate = opt.lr

    learning_rate_decay = LearningRateDecay(20, 1, 0.8)

    train_set_loader, test_set_loader = prepare_dataset(opt.bs, opt.fold)

    for epoch in range(start_epoch, TOTAL_EPOCH):
        state.set_epoch(epoch)
        state = train(epoch, state, learning_rate_decay, net, train_set_loader, learning_rate, optimizer, loss_fn)
        state = run_testing(epoch, state, net, test_set_loader, learning_rate, optimizer, loss_fn)
        plot_progress(state, f'{name}_{opt.fold}_{opt.bs}', img_path=google_drive_path)

    best_test_acc, best_test_epoch = state.get_best_test_acc()

    print("best_Test_acc: %0.3f" % best_test_acc)
    print("best_Test_acc_epoch: %d" % best_test_epoch)

