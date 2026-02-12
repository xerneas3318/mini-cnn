import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "MTCNN"))
from MTCNN import create_mtcnn_net  # noqa: E402


def load_minicnn(model_path, device, emb_dim=512, width_mult=0.75):
    minicnn_path = Path(__file__).resolve().parents[1] / "mini-cnn.py"
    spec = importlib.util.spec_from_file_location("minicnn_module", minicnn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.MiniCNN(emb_dim=emb_dim, width_mult=width_mult).to(device)
    model_path = resolve_path(model_path)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        if "model_state" in state:
            state_dict = state["model_state"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def resolve_path(path_str):
    path = Path(path_str)
    if path.exists():
        return path
    alt = REPO_ROOT / path_str
    if alt.exists():
        return alt
    return path


def resolve_output_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_detector(pnet_path, rnet_path, onet_path, device):
    pnet = resolve_path(pnet_path)
    rnet = resolve_path(rnet_path)
    onet = resolve_path(onet_path)
    for p in (pnet, rnet, onet):
        if not p.exists():
            raise FileNotFoundError(f"MTCNN weight not found: {p}")
    return {"pnet": str(pnet), "rnet": str(rnet), "onet": str(onet), "device": device}


def detect_faces(detector, image_bgr, min_face=32, scale=1.0):
    image = image_bgr
    if scale != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
    bboxes, landmarks = create_mtcnn_net(
        image,
        min_face,
        detector["device"],
        p_model_path=detector["pnet"],
        r_model_path=detector["rnet"],
        o_model_path=detector["onet"],
    )
    if bboxes is None or len(bboxes) == 0:
        return []
    landmarks = np.asarray(landmarks) if landmarks is not None else None
    out = []
    for i, b in enumerate(bboxes):
        box = b[:4].astype(float)
        score = float(b[4]) if len(b) > 4 else 1.0
        if scale != 1.0:
            box = box / scale
        lm = None
        if landmarks is not None and landmarks.size > 0:
            lm = landmarks[i].reshape(2, 5).T.astype(float)
            if scale != 1.0:
                lm = lm / scale
        out.append((box, score, lm))
    return out


def crop_face(image_bgr, box, pad=0.2):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad
    pad_y = bh * pad
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))
    return image_bgr[y1:y2, x1:x2]


def _transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    u, _, vt = np.linalg.svd(points1.T * points2)
    r = (u * vt).T
    return np.vstack(
        [
            np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)),
            np.matrix([0.0, 0.0, 1.0]),
        ]
    )


def align_face(image_bgr, landmarks, size=112):
    if landmarks is None or len(landmarks) == 0:
        return None
    coord5point = np.array(
        [
            [38.29459953, 51.69630051],
            [73.53179932, 51.50139999],
            [56.02519989, 71.73660278],
            [41.54930115, 92.3655014],
            [70.72990036, 92.20410156],
        ],
        dtype=np.float64,
    )
    pts1 = np.float64(np.matrix([[p[0], p[1]] for p in landmarks]))
    pts2 = np.float64(np.matrix([[p[0], p[1]] for p in coord5point]))
    m = _transformation_from_points(pts1, pts2)
    aligned = cv2.warpAffine(image_bgr, m[:2], (image_bgr.shape[1], image_bgr.shape[0]))
    return aligned[0:size, 0:size]


def preprocess_face(face_bgr, size=112):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_AREA)
    face = torch.from_numpy(face_rgb).float().permute(2, 0, 1) / 255.0
    face = (face - 0.5) / 0.5
    return face.unsqueeze(0)


def embed_face(model, face_tensor, device):
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        emb = model(face_tensor)
    return emb.squeeze(0).cpu()


def build_db(embeddings, labels):
    if len(embeddings) == 0:
        raise ValueError("No embeddings to build database.")
    emb = np.stack(embeddings, axis=0)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, np.array(labels)


def match_embedding(emb, db_emb, db_labels, threshold=0.4):
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    sims = np.dot(db_emb, emb)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return db_labels[best_idx], best_sim
    return "unknown", best_sim


def normalize_labels(labels, paths=None):
    if paths is None:
        return labels
    normalized = []
    for label, path in zip(labels, paths):
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        if label in {"images", "ipy_pic"} and path:
            label = Path(path).stem
        normalized.append(label)
    return np.array(normalized)
