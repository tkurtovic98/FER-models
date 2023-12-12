from models import get_model
from checkpoint import load_checkpoint, set_checkpoint_path
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch

import os

from sklearn.metrics import confusion_matrix

from dataset_loaders import get_dataset_loader

from torchvision import models

from main import parse_args

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


if __name__ == "__main__":
    opt = parse_args()
    
    if opt.dataset == "FER2013":
        class_names = ['Angry', 'Disgust', 'Fear',
                   'Happy', 'Sad', 'Surprise', 'Neutral']
    else:
        class_names = ['Angry', 'Disgust', 'Fear','Happy', 'Neutral','Sad', 'Surprise']

    root = opt.root 
    name = f'{opt.dataset if opt.outname is None else opt.outname}_{opt.model}'

    # if opt.pretrained:
    #     name+="_pretrained"

    path = os.path.join(root,name, opt.version)

    _, val_set_loader, test_set_loader =  get_dataset_loader(opt.dataset)(64)

    if opt.pretrained:
        if opt.model == "VGG19":
            net = models.vgg19(weights='DEFAULT')
            
            net.classifier = torch.nn.Linear(25088,7)
        elif opt.model == "AlexNet":
            net = torchmodels.alexnet(weights='DEFAULT')
            
            last_cls_in_features = (net.classifier[-1]).in_features
            
            new_last_cls = nn.Linear(last_cls_in_features, 7)
            
            net.classifier[-1] = new_last_cls
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

    correct = 0
    total = 0
    all_target = []

    CROPS_PRESENT_INPUT_DIMS = 5

    for batch_idx, (inputs, targets) in enumerate(test_set_loader):
        crops_present = len(np.shape(inputs)) == CROPS_PRESENT_INPUT_DIMS

        if crops_present:
            bs, ncrops, c, h, w = np.shape(inputs)
        else:
            bs, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)

        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        
        if not crops_present:
            outputs_avg = outputs
        else:
            outputs_avg = outputs.view(
                bs, ncrops, -1).mean(1)   # avg over crops
              
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
