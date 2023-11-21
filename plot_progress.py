import matplotlib.pyplot as plt

from training_state import TrainingState


def plot_progress(state: TrainingState):
    plt.figure(figsize=(12, 4))

    epochs = [i+1 for i in range(len(state.train_losses)) ]

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, state.train_losses, label='Training Loss')
    plt.plot(epochs, state.public_losses, label='Testing Loss')
    plt.plot(epochs, state.private_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Time - Current epoch {state.epoch}')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, state.train_accuracies, label='Training Accuracy')
    plt.plot(epochs, state.test_accuracies, label='Test Accuracy')
    plt.plot(epochs, state.validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()

    plt.show()
