"""
plot confusion_matrix of PublicTest and PrivateTest
"""

from models import get_model
from fer import prepare_dataset
from checkpoint import load_checkpoint, set_checkpoint_path
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch

import os
import argparse

from sklearn.metrics import confusion_matrix
from models import *


parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19',
                    help='CNN architecture')
parser.add_argument('--dataset', type=str,
                    default='FER2013', help='CNN architecture')
parser.add_argument('--split', type=str, default='Validation', help='split')
parser.add_argument('--pretrained', '-p', action='store_true', default=False)
opt = parser.parse_args()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

from torchvision import models


VERSION = "1.0"


if __name__ == "__main__":
    
    if opt.dataset == "FER2013":
        class_names = ['Angry', 'Disgust', 'Fear',
                   'Happy', 'Sad', 'Surprise', 'Neutral']
    else:
        class_names = ['Angry', 'Disgust', 'Fear','Happy', 'Neutral','Sad', 'Surprise']

    root = '/content/drive/MyDrive/FER_Doktorski/FER-models' if GOOGLE_DRIVE else './'
    name = f'{DATASET}_{MODEL}_pretrained'
    path = os.path.join(root,name, VERSION)
    
    if opt.pretrained:
        name+="_pretrained"
    
    path = os.path.join(google_drive_path, name)

    if opt.pretrained:
        if opt.model == "VGG19":
            net = models.vgg19(weights='DEFAULT')
            
            net.classifier = torch.nn.Linear(25088,7)
        else:
            net = models.resnet18(weights='DEFAULT')

            num_features = net.fc.in_features
            net.fc = torch.nn.Linear(num_features, 7)
    else:
        net = get_model(opt.model)

    set_checkpoint_path(path)

    checkpoint = load_checkpoint(opt.split + '_model.t7')

    net.load_state_dict(checkpoint.net)
    net.cuda()
    net.eval()

    _, _, test_set_loader = prepare_dataset(64)

    correct = 0
    total = 0
    all_target = []
    for batch_idx, (inputs, targets) in enumerate(test_set_loader):
        if len(np.shape(inputs)) == 5:
              bs, ncrops, c, h, w = np.shape(inputs)
        else:
          bs, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
       
        if len(np.shape(inputs)) != 5:
          outputs_avg = outputs
        else:
          outputs_avg = outputs.view(
              bs, ncrops, -1).mean(1)  # avg over crops  # avg over crops
              
        _, predicted = torch.max(outputs_avg.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx == 0:
            all_predicted = predicted
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, targets), 0)

    acc = 100. * correct / total
    print("accuracy: %0.3f" % acc)

    # Compute confusion matrix
    matrix = confusion_matrix(
        all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                          title=opt.split+' Confusion Matrix (Accuracy: %0.3f%%)' % acc)
    plt.savefig(os.path.join(path, opt.split + '_cm.png'))
    plt.close()
