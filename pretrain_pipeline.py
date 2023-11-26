import torch
from torchvision import models
from torchinfo import summary

from torch import nn, optim
from fer import prepare_dataset
from learning import LearningRateDecay, train, run_validation, run_testing
from plot_progress import plot_progress

import os

from training_state import TrainingState

from checkpoint import set_checkpoint_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOTAL_EPOCH = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.01
DATASET = "FER2013"
MODEL = "VGG19"
GOOGLE_DRIVE = False

if __name__ == "__main__":
    
    root = '/content/drive/MyDrive/FER_Doktorski/FER-models' if GOOGLE_DRIVE else './'
    name = f'{DATASET}_{MODEL}_pretrained' 
    path = os.path.join(root, name)

    set_checkpoint_path(path)
    
    state = TrainingState()

    net = models.vgg19(weights='DEFAULT')
    
    net.classifier = torch.nn.Linear(25088,7).to(device)
    
    print(net)

    for x in net.features.parameters():
        x.requires_grad = False
        
    train_set_loader, val_set_loader, test_set_loader = prepare_dataset(BATCH_SIZE)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    learning_rate_decay = LearningRateDecay(start = 120, every=5, rate=0.9)

    for epoch in range(0, TOTAL_EPOCH):
        state.set_epoch(epoch)
        state = train(epoch, state, learning_rate_decay, net, train_set_loader, LEARNING_RATE, optimizer, loss_fn)
        state = run_testing(epoch, state, net, test_set_loader, LEARNING_RATE, optimizer, loss_fn)
        state = run_validation(epoch, state, net, val_set_loader, LEARNING_RATE, optimizer, loss_fn)
        plot_progress(state)
