from pathlib import Path
import os
from typing import Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class IndexedDataset(Dataset):
    def __init__(self, index_file, transform=None):
        self.index_file = index_file
        self.transform = transform
        self.data = []
        self.labels = []
        self._label_to_idx = {}
        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def _load_data(self):
        raw_labels = []
        with open(self.index_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.data.append(path)
                raw_labels.append(int(label))
        unique = sorted(set(raw_labels))
        self._label_to_idx = {y: i for i, y in enumerate(unique)}
        self.labels = [self._label_to_idx[y] for y in raw_labels]


def read_lfw_pairs(pair_file, root_dir):
    pairs = []
    with open(pair_file, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()

            if len(parts) == 3 and parts[0].endswith(".jpg") and parts[1].endswith(".jpg"):
                p1, p2, label = parts
                p1 = Path(root_dir) / p1
                p2 = Path(root_dir) / p2
                pairs.append((str(p1), str(p2), int(label)))
                continue

            if len(parts) == 3:
                name, i1, i2 = parts
                f1 = i1 if i1.endswith(".jpg") else f"{int(i1):04d}.jpg"
                f2 = i2 if i2.endswith(".jpg") else f"{int(i2):04d}.jpg"
                p1 = Path(root_dir) / name / f"{name}_{f1}"
                p2 = Path(root_dir) / name / f"{name}_{f2}"
                pairs.append((str(p1), str(p2), 1))
            elif len(parts) == 4:
                name1, i1, name2, i2 = parts
                f1 = i1 if i1.endswith(".jpg") else f"{int(i1):04d}.jpg"
                f2 = i2 if i2.endswith(".jpg") else f"{int(i2):04d}.jpg"
                p1 = Path(root_dir) / name1 / f"{name1}_{f1}"
                p2 = Path(root_dir) / name2 / f"{name2}_{f2}"
                pairs.append((str(p1), str(p2), 0))
    return pairs


def read_pairs_from_file(pair_file, root_dir=None):
    pairs = []
    root_dir = Path(root_dir) if root_dir is not None else None
    with open(pair_file, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            p1, p2, label = line.strip().split()
            if root_dir is not None:
                if not os.path.isabs(p1):
                    p1 = root_dir / p1
                if not os.path.isabs(p2):
                    p2 = root_dir / p2
            pairs.append((str(p1), str(p2), int(label)))
    return pairs


def build_dataloaders(
    data_root: Path,
    batch_size: int = 192,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Dict[str, DataLoader], IndexedDataset, Dict[str, list]]:
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = IndexedDataset(
        data_root / "ms1m-arcface" / "index.txt",
        transform=train_transform,
    )

    import sys
    mfn_repo = Path("/home/xerneas/Coding/MobileFaceNet_Tutorial_Pytorch")
    sys.path.append(str(mfn_repo))
    from data_set.dataloader import LFW as MF_LFW, CFP_FP as MF_CFP_FP, AgeDB30 as MF_AgeDB30

    eval_transform = transforms.Compose([
        transforms.Lambda(lambda x: x[:, :, ::-1].copy()),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    lfw_dataset = MF_LFW(
        root=str(data_root / "LFW" / "lfw_align_112"),
        file_list=str(data_root / "LFW" / "pairs.txt"),
        transform=eval_transform,
    )
    cfp_dataset = MF_CFP_FP(
        root=str(data_root / "CFP-FP" / "CFP_FP_aligned_112"),
        file_list=str(data_root / "CFP-FP" / "cfp_fp_pair.txt"),
        transform=eval_transform,
    )
    agedb_dataset = MF_AgeDB30(
        root=str(data_root / "AgeDB-30" / "agedb30_align_112"),
        file_list=str(data_root / "AgeDB-30" / "agedb_30_pair.txt"),
        transform=eval_transform,
    )

    eval_loaders = {
        "LFW": DataLoader(lfw_dataset, batch_size=128, shuffle=False, num_workers=2, drop_last=False),
        "CFP-FP": DataLoader(cfp_dataset, batch_size=128, shuffle=False, num_workers=2, drop_last=False),
        "AgeDB-30": DataLoader(agedb_dataset, batch_size=128, shuffle=False, num_workers=2, drop_last=False),
    }

    pairs = {
        "LFW": read_lfw_pairs(
            pair_file=data_root / "LFW" / "pairs.txt",
            root_dir=data_root / "LFW" / "lfw_align_112",
        ),
        "CFP-FP": read_pairs_from_file(
            data_root / "CFP-FP" / "cfp_fp_pair.txt",
            root_dir=data_root / "CFP-FP" / "CFP_FP_aligned_112",
        ),
        "AgeDB-30": read_pairs_from_file(
            data_root / "AgeDB-30" / "agedb_30_pair.txt",
            root_dir=data_root / "AgeDB-30" / "agedb30_align_112",
        ),
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, eval_loaders, train_dataset, pairs
