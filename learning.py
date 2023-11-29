from __future__ import print_function
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import utils
from training_state import TrainingState
from checkpoint import Checkpoint, save_checkpoint

VALIDATION_MODEL_NAME = "Validation_model"
TEST_MODEL_NAME = "Test_model"

use_cuda = torch.cuda.is_available()


@dataclass
class LearningRateDecay():
    start: int
    every: int
    rate: float


def train(epoch, state: TrainingState, learning_rate_decay: LearningRateDecay, net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.CrossEntropyLoss()):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0

    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)

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

CROPS_PRESENT_INPUT_DIMS = 5

def _test(net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.CrossEntropyLoss()):
    net.eval()
    test_loss, correct, total = 0, 0, 0

    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            crops_present = len(np.shape(inputs)) == CROPS_PRESENT_INPUT_DIMS

            if crops_present:
              bs, ncrops, c, h, w = np.shape(inputs)
            else:
              bs, c, h, w = np.shape(inputs)

            inputs = inputs.view(-1, c, h, w)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            
            if not crops_present:
              outputs_avg = outputs
            else:
              outputs_avg = outputs.view(
                  bs, ncrops, -1).mean(1)  # avg over crops

            loss = loss_fn(outputs_avg, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    test_acc = 100. * correct / total

    return test_acc, test_loss/(batch_idx+1)


def run_testing(epoch, state: TrainingState, net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.CrossEntropyLoss()):
    test_acc, test_loss = _test(net, dataloader, lr, optimizer, loss_fn)

    if test_acc > state.get_best_test_acc()[0]:
        print('Saving..')
        print("Best test accuray: %0.3f" % test_acc)

        save_state = Checkpoint(
            net=net.state_dict(),
            best_test_acc=test_acc,
            best_test_epoch=epoch,
        )

        save_checkpoint(f'{TEST_MODEL_NAME}.t7', save_state)

    state.update_test_acc(test_acc, epoch)
    state.add_test_loss(test_loss)

    return state


def run_validation(epoch, state: TrainingState, net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.CrossEntropyLoss()):
    validation_acc, test_loss = _test(net, dataloader, lr, optimizer, loss_fn)

    if validation_acc > state.get_best_validation_acc()[0]:
        print('Saving..')
        print("Best validation test accuracy: %0.3f" % validation_acc)

        best_test_acc, best_test_epoch = state.get_best_test_acc()

        save_state = Checkpoint(
            net=net.state_dict(),
            best_test_acc=best_test_acc,
            best_test_epoch=best_test_epoch,
            best_val_acc=validation_acc,
            best_val_epoch=epoch
        )

        save_checkpoint(f'{VALIDATION_MODEL_NAME}.t7', save_state)

    state.update_validation_acc(validation_acc, epoch)
    state.add_validation_loss(test_loss)

    return state

