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

# label_to_weight: dict[int, float] = {
#     0: 7215/3995,
#     1: 7215/436,
#     2: 7215/4097,
#     3: 1,
#     4: 7215/4830,
#     5: 7215/3171,
#     6: 7215/4965
# }

label_to_weight: dict[int, float] = {
    0: 193/171,
    1: 193/142,
    2: 1,
    3: 193/83,
    4: 193/154,
    5: 193/62,
    6: 193/91
}


from main import parse_args

if __name__ == "__main__":
    opt = parse_args()

    root_path = opt.root 
    name = f'{opt.dataset}_{opt.model}'
    path = os.path.join(root_path, name, opt.version)

    set_checkpoint_path(path)

    state = TrainingState()

    if opt.model == "VGG19":
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
    if opt.resume:
        checkpoint = load_checkpoint('Validation_model.t7')

        net.load_state_dict(checkpoint.net)

        state.best_test_acc = checkpoint.best_test_acc
        state.best_test_acc_epoch = checkpoint.best_test_epoch

        state.best_validation_acc = checkpoint.best_val_acc
        state.best_validation_acc_epoch = checkpoint.best_val_epoch

        start_epoch = checkpoint.best_val_epoch + 1

    net.cuda()

    print(net)

    train_set_loader, val_set_loader, test_set_loader = get_dataset_loader(opt.dataset)(opt.bs)

    weights = torch.from_numpy(np.array(list(label_to_weight.values()), dtype=np.float32)).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    learning_rate = opt.lr
    learning_rate_decay = LearningRateDecay(start = opt.lds, every=opt.lde, rate=opt.ldr)

    for epoch in range(start_epoch, opt.epoch):
        state.set_epoch(epoch)
        state = train(epoch, state, learning_rate_decay, net,
                      train_set_loader, learning_rate, optimizer, loss_fn)
        state = run_validation(
            epoch, state, net, val_set_loader, learning_rate, optimizer, loss_fn)
        plot_progress(state, name, img_path=path)

    best_validation_acc, best_validation_epoch = state.get_best_validation_acc()

    print(
        f"Best Public Test Accuracy: {best_validation_acc} at Epoch {best_validation_epoch}")
