import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from face_utils import (
    crop_face,
    detect_faces,
    embed_face,
    align_face,
    load_detector,
    load_minicnn,
    match_embedding,
    normalize_labels,
    preprocess_face,
    resolve_path,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="tester/embeddings_rian.npz")
    parser.add_argument("--pnet", default="MTCNN/weights/pnet_Weights")
    parser.add_argument("--rnet", default="MTCNN/weights/rnet_Weights")
    parser.add_argument("--onet", default="MTCNN/weights/onet_Weights")
    parser.add_argument("--min-face", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--model", required=True, help="Face recognition model path (.pt/.pth)")
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument("--width-mult", type=float, default=0.75)
    parser.add_argument("--pad", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Require best - second_best >= margin to accept",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Use MTCNN landmark alignment before embedding",
    )
    parser.add_argument("--camera", type=int, default=1)
    args = parser.parse_args()

    data = np.load(resolve_path(args.embeddings), allow_pickle=True)
    db_emb = data["embeddings"]
    db_labels = data["labels"]
    db_paths = data["paths"] if "paths" in data else None
    db_labels = normalize_labels(db_labels, db_paths)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = load_detector(args.pnet, args.rnet, args.onet, device)
    model = load_minicnn(args.model, device, emb_dim=args.emb_dim, width_mult=args.width_mult)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(detector, frame, min_face=args.min_face, scale=args.scale)
        for box, score, lm in faces:
            face = None
            if args.align and lm is not None:
                face = align_face(frame, lm)
            if face is None:
                face = crop_face(frame, box, pad=args.pad)
            if face.size == 0:
                continue
            face_tensor = preprocess_face(face)
            emb = embed_face(model, face_tensor, device).numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            sims = np.dot(db_emb, emb)
            if len(sims) == 0:
                label, sim = "unknown", 0.0
            else:
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if len(sims) > 1:
                    second_sim = float(np.partition(sims, -2)[-2])
                else:
                    second_sim = -1.0
                if best_sim >= args.threshold and (best_sim - second_sim) >= args.margin:
                    label, sim = db_labels[best_idx], best_sim
                else:
                    label, sim = "unknown", best_sim
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {sim:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("live-rec", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
