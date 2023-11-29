import os
import torch
from torch import nn, optim

from mainpro_FER import parse_args
from checkpoint import set_checkpoint_path, load_checkpoint
from models import get_model
from training_state import TrainingState
from learning import LearningRateDecay, run_validation, train
from plot_progress import plot_progress
from SFEW import prepare_dataset

TOTAL_EPOCH = 250

if __name__ == "__main__":
    opt = parse_args()
    
    assert opt.dataset == "SFEW", "Dataset should be SFEW"
    
    google_drive_path = '/content/drive/MyDrive/FER_Doktorski/FER-models'
    name = opt.dataset + '_' + opt.model
    path = os.path.join(google_drive_path, name)

    set_checkpoint_path(path)

    net = get_model(opt.model)

    state = TrainingState()

    use_cuda = torch.cuda.is_available()

    if opt.resume:
        # Load checkpoint.
        checkpoint = load_checkpoint('PrivateTest_model.t7')

        net.load_state_dict(checkpoint.net)

        state.best_test_acc = checkpoint.best_test_acc
        state.best_test_acc_epoch = checkpoint.best_test_epoch

        state.best_validation_acc = checkpoint.best_val_acc
        state.best_validation_acc_epoch = checkpoint.best_val_epoch

        start_epoch = checkpoint.best_val_epoch + 1
    else:
        print('==> Building model..')
        start_epoch = 0

    if use_cuda:
        net.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    learning_rate = opt.lr
    learning_rate_decay = LearningRateDecay(start = 100, every=5, rate=0.9)

    train_set_loader, validation_set_loader, test_set_loader = prepare_dataset(opt.bs)
    
    for epoch in range(start_epoch, TOTAL_EPOCH):
        state.set_epoch(epoch)
        state = train(epoch, state, learning_rate_decay, net, train_set_loader, learning_rate, optimizer, loss_fn)
        state = run_validation(epoch, state, net, validation_set_loader, learning_rate, optimizer, loss_fn)
        plot_progress(state, name, img_path = google_drive_path)

    best_validation_acc, best_validation_epoch = state.get_best_validation_acc()

    print(
        f"Best Validation Test Accuracy: {best_validation_acc} at Epoch {best_validation_epoch}")

    