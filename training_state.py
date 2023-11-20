class TrainingState:
    def __init__(self):
        self.best_PublicTest_acc = 0
        self.best_PublicTest_acc_epoch = 0
        self.best_PrivateTest_acc = 0
        self.best_PrivateTest_acc_epoch = 0
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.validation_accuracies = []
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def add_train_acc(self, acc):
        self.train_accuracies.append(acc)

    def add_train_loss(self, loss):
        self.train_losses.append(loss)

    def update_PublicTest_acc(self, acc, epoch):
        if self.best_PublicTest_acc != 0:
            self.test_accuracies.append(acc)
        if acc > self.best_PublicTest_acc:
            self.best_PublicTest_acc = acc
            self.best_PublicTest_acc_epoch = epoch

    def update_PrivateTest_acc(self, acc, epoch):
        if self.best_PrivateTest_acc != 0:
            self.validation_accuracies.append(acc)
        if acc > self.best_PrivateTest_acc:
            self.best_PrivateTest_acc = acc
            self.best_PrivateTest_acc_epoch = epoch

    def get_best_PublicTest_acc(self):
        return self.best_PublicTest_acc, self.best_PublicTest_acc_epoch

    def get_best_PrivateTest_acc(self):
        return self.best_PrivateTest_acc, self.best_PrivateTest_acc_epoch
