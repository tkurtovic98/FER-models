import os
import torch
from torchvision import models
from torch import nn, optim

import numpy as np

from dataset_loaders import get_dataset_loader
from learning import LearningRateDecay, train, run_validation
from plot_progress import plot_progress


from training_state import TrainingState

from checkpoint import set_checkpoint_path, load_checkpoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOTAL_EPOCH = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.01
DATASET = "FER2013"
MODEL = "Resnet18"
GOOGLE_DRIVE = True
RESUME = False
VERSION = "1.0"


prepare_dataset = get_dataset_loader(DATASET)

label_to_weight: dict[int, float] = {
    0: 7215/3995,
    1: 7215/436,
    2: 7215/4097,
    3: 1,
    4: 7215/4830,
    5: 7215/3171,
    6: 7215/4965
}

if __name__ == "__main__":
    root = '/content/drive/MyDrive/FER_Doktorski/FER-models' if GOOGLE_DRIVE else './'
    name = f'{DATASET}_{MODEL}_pretrained'
    path = os.path.join(root,name, VERSION)

    set_checkpoint_path(path)

    state = TrainingState()

    if MODEL == "VGG19":
        net = models.vgg19(weights='DEFAULT')

        net.classifier = torch.nn.Linear(25088, 7).to(device)

        for x in net.features.parameters():
            x.requires_grad = False

    else:
        net = models.resnet18(weights='DEFAULT')

        num_features = net.fc.in_features
        net.fc = torch.nn.Linear(num_features, 7).to(device)

        net.conv1.requires_grad = False
        net.bn1.requires_grad = False
        net.layer1.requires_grad = False
        net.layer2.requires_grad = False
        net.layer3.requires_grad = False
        net.layer4.requires_grad = False

    start_epoch = 0
    if RESUME:
        checkpoint = load_checkpoint('PrivateTest_model.t7')

        net.load_state_dict(checkpoint.net)

        state.best_test_acc = checkpoint.best_test_acc
        state.best_test_acc_epoch = checkpoint.best_test_epoch

        state.best_validation_acc = checkpoint.best_val_acc
        state.best_validation_acc_epoch = checkpoint.best_val_epoch

        start_epoch = checkpoint.best_val_epoch + 1

    net.cuda()

    print(net)

    train_set_loader, val_set_loader, test_set_loader = prepare_dataset(
        BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(list(label_to_weight.values()), dtype=np.float32)))
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE,
                          momentum=0.9, weight_decay=5e-4)
    learning_rate_decay = LearningRateDecay(start=60, every=5, rate=0.9)

    for epoch in range(start_epoch, TOTAL_EPOCH):
        state.set_epoch(epoch)
        state = train(epoch, state, learning_rate_decay, net,
                      train_set_loader, LEARNING_RATE, optimizer, loss_fn)
        state = run_validation(
            epoch, state, net, val_set_loader, LEARNING_RATE, optimizer, loss_fn)
        plot_progress(state, name, img_path=root)

    best_validation_acc, best_validation_epoch = state.get_best_validation_acc()

    print(
        f"Best Public Test Accuracy: {best_validation_acc} at Epoch {best_validation_epoch}")
