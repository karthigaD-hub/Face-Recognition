# gallery_build.py
import torch
import os
import cv2
import numpy as np
import pickle
from model import FaceNet
from utils import preprocess_face

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceNet().to(device)
model.eval()

gallery_dir = "gallery"
os.makedirs(gallery_dir, exist_ok=True)

gallery = {}

for person_name in os.listdir(gallery_dir):
    img_path = os.path.join(gallery_dir, person_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    face_tensor = preprocess_face(img).to(device)
    emb = model(face_tensor).detach().cpu()
    gallery[os.path.splitext(person_name)[0]] = emb

with open("gallery_embeddings.pkl", "wb") as f:
    pickle.dump(gallery, f)

print("✅ Gallery built successfully!")
