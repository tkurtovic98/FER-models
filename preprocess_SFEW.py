import numpy as np
import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    device=device,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7]
)


def normalize(image: Image) -> Image:
    img_arr = np.array(image)

    minval = img_arr.min()
    maxval = img_arr.max()
    if minval != maxval:
        img_arr -= minval
        img_arr = img_arr * (255/(maxval-minval))

    norm_image = Image.fromarray(img_arr.astype('uint8'), image.mode)

    return norm_image


def crop_face(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')

    # Detect faces
    boxes, probs = mtcnn.detect(image)

    # Crop the first face found with a high enough confidence
    if boxes is not None and probs is not None:
        for box, prob in zip(boxes, probs):
            if prob > 0.9:  # Adjust this threshold as needed
                box = box.astype(int)
                face = image.crop(box)
                face = face.convert('L')
                return normalize(face)
    return None


def resize_image(face, size):
    face_resized = face.resize(size)
    return face_resized


def preprocess_and_save(input_dir, output_dir, target_size=(224, 224)):
    dataset_types = os.listdir(input_dir)
    for dataset_type in dataset_types:
        if not os.path.isdir(os.path.join(input_dir, dataset_type)):
            continue
        if dataset_type not in ["Train", "Val"]:
            continue
        dataset_path = os.path.join(input_dir, dataset_type)
        output_dataset_path = os.path.join(output_dir, dataset_type)
        os.makedirs(output_dataset_path, exist_ok=True)

        emotions = os.listdir(dataset_path)
        for emotion in emotions:
            emotion_dir = os.path.join(dataset_path, emotion)
            output_emotion_dir = os.path.join(output_dataset_path, emotion)
            os.makedirs(output_emotion_dir, exist_ok=True)

            for img_file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_file)
                face = crop_face(img_path)

                if face is not None:
                    face = resize_image(face, target_size)
                    face.save(os.path.join(output_emotion_dir, img_file))


# Paths to your dataset
INPUT_DIR = './SFEW'  # Update this path
OUTPUT_DIR = './data/SFEW'  # Update this path

# Preprocess and save images
preprocess_and_save(INPUT_DIR, OUTPUT_DIR, (256, 256))
