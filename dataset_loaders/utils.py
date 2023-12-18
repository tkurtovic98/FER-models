import matplotlib.pyplot as plt

def _features_and_labels(dataloader, split):
    t, v, _ = dataloader()

    loader = t if split == "Train" else v

    features, labels = next(iter(loader))

    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")

    return features, labels


def plot_single_image(dataloader, split="Train", position = 0):
    features, labels = _features_and_labels(dataloader, split)

    img = features[position].squeeze()
    img = img.squeeze()
    plt.imshow(img.permute(1,2,0))

    plt.show()


def plot_multiple_images(dataloader, split="Train", how_many=5):
    for i in range(how_many):
        plot_single_image(dataloader, split, i)

import datetime

def plot_ten_crop_images(dataloader, split="Train", how_many=3, save=False):
    features,labels = _features_and_labels(dataloader, split)

    plt.figure(figsize=(12, 4))

    for j in range(how_many):
        for i, img in enumerate(features[j]):
            plt.subplot(2, 5, i+1)
            img = img.squeeze()
            plt.imshow(img.permute(1,2,0))

        if save:
            plt.savefig(f'{datetime.datetime.now()}_{j}.png')
        else:
            plt.show(block=True)
