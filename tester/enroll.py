import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from face_utils import (
    build_db,
    crop_face,
    detect_faces,
    embed_face,
    align_face,
    load_detector,
    load_minicnn,
    preprocess_face,
    resolve_output_path,
    resolve_path,
)


def iter_images(root):
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="rian")
    parser.add_argument("--pnet", default="MTCNN/weights/pnet_Weights")
    parser.add_argument("--rnet", default="MTCNN/weights/rnet_Weights")
    parser.add_argument("--onet", default="MTCNN/weights/onet_Weights")
    parser.add_argument("--min-face", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--model", required=True, help="Face recognition model path (.pt/.pth)")
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument("--width-mult", type=float, default=0.75)
    parser.add_argument("--out", default="tester/embeddings_rian.npz")
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument(
        "--single-label",
        default="rian",
        help="If set, use this label for all images",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Use MTCNN landmark alignment before embedding",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = load_detector(args.pnet, args.rnet, args.onet, device)
    model = load_minicnn(args.model, device, emb_dim=args.emb_dim, width_mult=args.width_mult)

    embeddings = []
    labels = []
    paths = []

    images_dir = resolve_path(args.images_dir)
    for path in tqdm(list(iter_images(images_dir)), desc="enroll"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = detect_faces(detector, img, min_face=args.min_face, scale=args.scale)
        if not faces:
            continue
        box, _, lm = max(faces, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
        face = None
        if args.align and lm is not None:
            face = align_face(img, lm)
        if face is None:
            face = crop_face(img, box, pad=args.pad)
        face_tensor = preprocess_face(face)
        emb = embed_face(model, face_tensor, device).numpy()
        # If images are directly under images-dir, use filename stem as label.
        # If images are in subfolders, use the top-level subfolder name.
        if args.single_label:
            label = args.single_label
        else:
            images_root = Path(images_dir).resolve()
            try:
                rel = path.resolve().relative_to(images_root)
                if rel.parent == Path("."):
                    label = path.stem
                else:
                    label = rel.parts[0]
            except ValueError:
                label = path.parent.name
        embeddings.append(emb)
        labels.append(label)
        paths.append(str(path))

    if not embeddings:
        raise SystemExit(
            "No embeddings created. Check --images-dir has face images and detector is working."
        )
    if args.single_label:
        # Average all embeddings into one template for a single identity.
        embeddings = [np.mean(np.stack(embeddings, axis=0), axis=0)]
        labels = [args.single_label]
        paths = [str(images_dir)]
    db_emb, db_labels = build_db(embeddings, labels)
    out_path = resolve_output_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=db_emb, labels=db_labels, paths=np.array(paths))
    print(f"Saved {len(db_labels)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
