import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from face_utils import align_face, crop_face, detect_faces, load_detector, resolve_path


def iter_images(root):
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="tester/images")
    parser.add_argument("--out-dir", default="tester/crops")
    parser.add_argument("--pnet", default="MTCNN/weights/pnet_Weights")
    parser.add_argument("--rnet", default="MTCNN/weights/rnet_Weights")
    parser.add_argument("--onet", default="MTCNN/weights/onet_Weights")
    parser.add_argument("--min-face", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument(
        "--align",
        action="store_true",
        help="Use MTCNN landmark alignment before saving crops",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = load_detector(args.pnet, args.rnet, args.onet, device)

    images_dir = resolve_path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for path in tqdm(list(iter_images(images_dir)), desc="crop"):
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue
        faces = detect_faces(detector, img, min_face=args.min_face, scale=args.scale)
        if not faces:
            skipped += 1
            continue
        box, _, lm = max(faces, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
        face = None
        if args.align and lm is not None:
            face = align_face(img, lm)
        if face is None:
            face = crop_face(img, box, pad=args.pad)
        if face.size == 0:
            skipped += 1
            continue
        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), face)
        saved += 1

    print(f"Saved {saved} crops to {out_dir}")
    print(f"Skipped {skipped} images")


if __name__ == "__main__":
    main()
