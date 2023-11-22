from __future__ import print_function
from plot_progress import plot_progress
from training_state import TrainingState

import torch
import numpy as np
import os
import utils
import torch.nn as nn

from dataclasses import dataclass

from checkpoint import  Checkpoint, save_checkpoint

@dataclass
class LearningRateDecay():
    start: int
    every: int
    rate: float

def train(epoch, state: TrainingState, learning_rate_decay: LearningRateDecay, use_cuda, net,dataloader,lr=0.01,optimizer=None,loss_fn = nn.CrossEntropyLoss()):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0,0,0

    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)

    if epoch > learning_rate_decay.start and learning_rate_decay.start >= 0:
        frac = (epoch - learning_rate_decay.start) // learning_rate_decay.every
        decay_factor = learning_rate_decay.rate ** frac
        current_lr = lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc = 100. * correct / total

    state.add_train_acc(train_acc)
    state.add_train_loss(train_loss/(batch_idx+1))

    return state

use_cuda = torch.cuda.is_available()

def _test(net,dataloader,lr=0.01,optimizer=None,loss_fn = nn.CrossEntropyLoss()):
    net.eval()
    test_loss, correct, total = 0,0,0

    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            
            loss = loss_fn(outputs_avg, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_acc = 100. * correct / total

    return test_acc, test_loss/(batch_idx+1)


def run_testing(epoch, state: TrainingState, net,dataloader,lr=0.01,optimizer=None,loss_fn = nn.CrossEntropyLoss()):
    test_acc, test_loss = _test(net, dataloader, lr, optimizer, loss_fn)

    if test_acc > state.get_best_PublicTest_acc()[0]:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % test_acc)

        save_state = Checkpoint(
            net= net.state_dict(),
            best_test_acc= test_acc,
            best_test_epoch= epoch,
        )

        save_checkpoint('PublicTest_model.t7', save_state)

    state.update_PublicTest_acc(test_acc, epoch)
    state.add_public_test_loss()

    return state

def run_validation(epoch, state: TrainingState, net,dataloader,lr=0.01,optimizer=None,loss_fn = nn.CrossEntropyLoss()):
    test_acc, test_loss = _test(net, dataloader, lr, optimizer, loss_fn)

    if test_acc > state.get_best_PrivateTest_acc()[0]:
            print('Saving..')
            print("best_PrivateTest_acc: %0.3f" % test_acc)

            best_public_acc, best_public_epoch = state.get_best_PublicTest_acc()

            save_state = Checkpoint(
                net= net.state_dict(),
                best_test_acc=best_public_acc,
                best_test_epoch=best_public_epoch,
                best_val_acc=test_acc,
                best_val_epoch=epoch
            )

            save_checkpoint('PrivateTest_model.t7', save_state)

    state.update_PrivateTest_acc(test_acc, epoch)
    state.add_private_test_loss(test_loss) 

    return state
    