import torch, argparse, os, numpy as np
from model import FaceNet
from utils import load_and_align, compute_cosine, device
from PIL import Image, ImageDraw, ImageFont
import glob
import matplotlib.pyplot as plt

def extract_embeddings(model, gallery_dir, save_path):
    model.eval()
    classes = sorted(os.listdir(gallery_dir))
    gallery_embs, gallery_labels = [], []

    for c in classes:
        imgs = glob.glob(os.path.join(gallery_dir, c, "*.jpg"))
        for img in imgs:
            x = load_and_align(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb, _, _ = model(x)
            gallery_embs.append(emb.cpu().numpy().flatten())
            gallery_labels.append(c)

    np.savez(save_path, emb=np.array(gallery_embs), labels=np.array(gallery_labels))
    print(f"✅ Saved gallery embeddings at {save_path}")


def infer(model, gallery_emb, gallery_labels, img_path, topk=3, show=True, save_dir="./out/predictions"):
    os.makedirs(save_dir, exist_ok=True)

    # handle folder or single image
    if os.path.isdir(img_path):
        img_list = glob.glob(os.path.join(img_path, "*.jpg")) + \
                   glob.glob(os.path.join(img_path, "*.png")) + \
                   glob.glob(os.path.join(img_path, "*.jpeg"))
    else:
        img_list = [img_path]

    for img_file in img_list:
        x = load_and_align(img_file).unsqueeze(0).to(device)
        with torch.no_grad():
            emb, _, _ = model(x)
        emb = emb.cpu().numpy().flatten()

        sims = [compute_cosine(emb, g) for g in gallery_emb]
        topk_idx = np.argsort(sims)[::-1][:topk]
        results = [(gallery_labels[i], sims[i]) for i in topk_idx]

        print(f"\n🔍 Top identity matches for {os.path.basename(img_file)}:")
        for name, s in results:
            print(f" {name} ({s:.3f})")

        # visualization
        try:
            img = Image.open(img_file).convert("RGB")
            draw = ImageDraw.Draw(img)
            text = f"Predicted: {results[0][0]} ({results[0][1]:.3f})"
            draw.rectangle([(0, 0), (img.width, 30)], fill=(0, 0, 0, 180))
            draw.text((10, 5), text, fill=(255, 255, 255))

            # save image with prediction
            save_path = os.path.join(save_dir, os.path.basename(img_file))
            img.save(save_path)

            # show image
            if show:
                plt.imshow(img)
                plt.axis("off")
                plt.title(text)
                plt.show()

            print(f"✅ Saved prediction: {save_path}")
        except Exception as e:
            print(f"⚠️ Could not display/save image {img_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['extract', 'infer'], required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--gallery_dir', type=str)
    parser.add_argument('--gallery_emb', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--save_path', type=str, default='./out/gallery_embeddings.npz')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--no_show', action='store_true', help="Disable image display (for servers)")
    args = parser.parse_args()

    model = FaceNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if args.mode == 'extract':
        extract_embeddings(model, args.gallery_dir, args.save_path)
    elif args.mode == 'infer':
        data = np.load(args.gallery_emb, allow_pickle=True)
        infer(model, data['emb'], data['labels'], args.img_path, args.topk, show=not args.no_show)
