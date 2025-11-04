import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2, os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)

transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def load_and_align(img_path):
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        img = transform(img)
        return img
    return face

def compute_cosine(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)
