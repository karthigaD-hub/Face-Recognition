import torch, os, argparse, numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from model import FaceNet
from utils import transform, device

def train(args):
    train_ds = datasets.ImageFolder(args.data_dir, transform=transform)
    val_ds = datasets.ImageFolder(args.val_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FaceNet(num_classes=len(train_ds.classes)).to(device)
    criterion_id = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    FEMALE_NAMES = ["Anushka", "Kajol"]
    MALE_NAMES = ["Vijay", "Ajith", "Rajini", "Kamal"]

    print("🚀 Training started...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct_gender = 0
        total_samples = 0

        for img, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            img, labels = img.to(device), labels.to(device)

            genders = []
            for l in labels:
                name = train_ds.classes[l]
                if any(n.lower() in name.lower() for n in FEMALE_NAMES):
                    genders.append(1)  # Female = 1
                else:
                    genders.append(0)  # Male = 0
            gender = torch.tensor(genders).to(device)

            emb, id_out, g_out = model(img)
            loss_id = criterion_id(id_out, labels)
            loss_gender = criterion_gender(g_out, gender)
            loss = loss_id + 0.4 * loss_gender

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred_gender = g_out.argmax(dim=1)
            correct_gender += (pred_gender == gender).sum().item()
            total_samples += gender.size(0)

        gender_acc = 100 * correct_gender / total_samples
        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {total_loss/len(train_loader):.4f} | Gender Acc: {gender_acc:.2f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "checkpoint_best.pth"))
    np.save(os.path.join(args.save_dir, "classes.npy"), train_ds.classes)
    print("✅ Model and classes saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./out')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    train(args)
