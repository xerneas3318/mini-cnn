import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from face_utils import load_minicnn, resolve_path


class IndexedDataset:
    def __init__(self, index_file, transform=None):
        self.index_file = Path(index_file)
        self.transform = transform
        self.data = []
        self.labels = []
        self._label_to_idx = {}
        self._load()

    def _load(self):
        raw_labels = []
        with open(self.index_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.data.append(path)
                raw_labels.append(int(label))
        unique = sorted(set(raw_labels))
        self._label_to_idx = {y: i for i, y in enumerate(unique)}
        self.labels = [self._label_to_idx[y] for y in raw_labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


@torch.no_grad()
def collapse_check(dataset, model, device, n_same=200, n_diff=200, seed=123):
    rng = np.random.default_rng(seed)
    label_to_indices = defaultdict(list)
    for i, y in enumerate(dataset.labels):
        label_to_indices[y].append(i)
    labels = list(label_to_indices.keys())

    same_sims = []
    diff_sims = []

    def embed_idx(i):
        img, _ = dataset[i]
        emb = model(img.unsqueeze(0).to(device))
        return emb.cpu()

    while len(same_sims) < n_same:
        y = rng.choice(labels)
        idxs = label_to_indices[y]
        if len(idxs) < 2:
            continue
        i1, i2 = rng.choice(idxs, size=2, replace=False)
        e1 = embed_idx(i1)
        e2 = embed_idx(i2)
        same_sims.append(float(torch.nn.functional.cosine_similarity(e1, e2).item()))

    while len(diff_sims) < n_diff:
        y1, y2 = rng.choice(labels, size=2, replace=False)
        i1 = rng.choice(label_to_indices[y1])
        i2 = rng.choice(label_to_indices[y2])
        e1 = embed_idx(i1)
        e2 = embed_idx(i2)
        diff_sims.append(float(torch.nn.functional.cosine_similarity(e1, e2).item()))

    same_sims = np.array(same_sims)
    diff_sims = np.array(diff_sims)

    print("same mean:", same_sims.mean(), "same std:", same_sims.std())
    print("diff mean:", diff_sims.mean(), "diff std:", diff_sims.std())
    print("same min/max:", same_sims.min(), same_sims.max())
    print("diff min/max:", diff_sims.min(), diff_sims.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", required=True, help="index.txt with 'path label' per line")
    parser.add_argument("--model", required=True, help="model checkpoint (.pt/.pth)")
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument("--width-mult", type=float, default=0.75)
    parser.add_argument("--n-same", type=int, default=200)
    parser.add_argument("--n-diff", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = IndexedDataset(resolve_path(args.index_file), transform=transform)
    model = load_minicnn(args.model, args.device, emb_dim=args.emb_dim, width_mult=args.width_mult)
    model.eval()

    collapse_check(
        dataset,
        model,
        device=args.device,
        n_same=args.n_same,
        n_diff=args.n_diff,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
