import math
import time
from pathlib import Path

import importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from dataloader import build_dataloaders
from torchao import quantize_
from torchao.quantization.quant_api import Int8WeightOnlyConfig


def load_minicnn(emb_dim=512, width_mult=0.75):
    minicnn_path = Path(__file__).resolve().parent / "mini-cnn.py"
    spec = importlib.util.spec_from_file_location("minicnn_module", minicnn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniCNN(emb_dim=emb_dim, width_mult=width_mult), module


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    return torch.div(input, norm)


class Arcface(nn.Module):
    def __init__(self, embedding_size=512, classnum=51332, s=64.0, m=0.5):
        super().__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        nn.init.xavier_uniform_(self.kernel)
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m
        self.threshold = math.cos(math.pi - m)

    def forward(self, embeddings, label):
        nB = len(embeddings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long, device=embeddings.device)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s
        return output


def getAccuracy(scores, flags, threshold, method):
    if method == "l2_distance":
        p = np.sum(scores[flags == 1] < threshold)
        n = np.sum(scores[flags == -1] > threshold)
    elif method == "cos_distance":
        p = np.sum(scores[flags == 1] > threshold)
        n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum, method):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 3.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i], method)
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def getFeature_mfn(net, dataloader, device, flip=True):
    featureLs = None
    featureRs = None

    for det in dataloader:
        for i in range(len(det)):
            det[i] = det[i].to(device)

        with torch.no_grad():
            res = [net(d).data.cpu() for d in det]

        if flip:
            featureL = l2_norm(res[0] + res[1])
            featureR = l2_norm(res[2] + res[3])
        else:
            featureL = res[0]
            featureR = res[2]

        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = torch.cat((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = torch.cat((featureRs, featureR), 0)

    return featureLs, featureRs


def evaluation_10_fold_mfn(featureL, featureR, dataset, method="l2_distance"):
    ACCs = np.zeros(10)
    threshold = np.zeros(10)
    fold = np.array(dataset.folds).reshape(1, -1)
    flags = np.array(dataset.flags).reshape(1, -1)
    flags_1d = np.squeeze(flags)

    featureL_np = featureL.numpy() if hasattr(featureL, "numpy") else np.asarray(featureL)
    featureR_np = featureR.numpy() if hasattr(featureR, "numpy") else np.asarray(featureR)

    for i in range(10):
        valFold = (fold != i).ravel()
        testFold = (fold == i).ravel()

        featureLs = featureL_np.copy()
        featureRs = featureR_np.copy()

        mu = np.mean(np.concatenate((featureLs[valFold, :], featureRs[valFold, :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        if method == "l2_distance":
            scores = np.sum(np.power((featureLs - featureRs), 2), 1)
        elif method == "cos_distance":
            scores = np.sum(np.multiply(featureLs, featureRs), 1)

        threshold[i] = getThreshold(scores[valFold], flags_1d[valFold], 10000, method)
        ACCs[i] = getAccuracy(scores[testFold], flags_1d[testFold], threshold[i], method)

    return ACCs, threshold


def main():
    emb_dim = 512
    num_epochs = 20
    verif_interval = 5000
    train_log_interval = 1000
    use_torchao_int8 = True

    data_root = Path("~/Datasets").expanduser()
    train_loader, eval_loaders, train_dataset, _pairs = build_dataloaders(data_root)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    log_dir = Path("logs")
    train_log_dir = log_dir / "train"
    test_log_dir = log_dir / "test"
    train_log_dir.mkdir(parents=True, exist_ok=True)
    test_log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    train_log_path = train_log_dir / f"train_{run_id}.log"
    test_log_path = test_log_dir / f"test_{run_id}.log"

    def write_train(msg):
        with open(train_log_path, "a") as f:
            f.write(msg + "\n")

    def write_test(msg):
        with open(test_log_path, "a") as f:
            f.write(msg + "\n")

    num_classes = len(set(train_dataset.labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, minicnn_module = load_minicnn(emb_dim=emb_dim)
    model = model.to(device)
    head = Arcface(embedding_size=emb_dim, classnum=num_classes, s=64.0, m=0.5).to(device)

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=0.01,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5, 8, 10],
        gamma=0.3,
    )

    start_epoch = 0
    global_step = 0
    resume_loaded = False

    if resume_loaded:
        print(f"[TRAIN] RESUMING from epoch={start_epoch}, global_step={global_step}")
    else:
        print("[TRAIN] STARTING fresh from epoch=0")
    write_train(f"[RUN] id={run_id} epochs={num_epochs} emb_dim={emb_dim} verif_interval={verif_interval}")
    write_train(f"[RUN] train_log_interval={train_log_interval} num_classes={num_classes}")
    write_train(f"[RUN] train_log={train_log_path} test_log={test_log_path}")
    write_train(f"[RUN] torchao_int8={use_torchao_int8}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        head.train()

        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(imgs)
            logits = head(embeddings, labels)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=1.0
            )
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                emb_norm = F.normalize(embeddings, dim=1)
                W_norm = F.normalize(head.kernel, dim=0)
                cos_logits = emb_norm @ W_norm
                preds = torch.argmax(cos_logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            global_step += 1
            if global_step % train_log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                step_msg = (
                    f"step={global_step} epoch={epoch+1} "
                    f"loss={running_loss / max(1, pbar.n):.4f} "
                    f"acc={correct / max(1, total):.6f} lr={lr}"
                )
                write_train(step_msg)
            if global_step % verif_interval == 0:
                pbar.clear()
                pbar.disable = True
                verif_start = time.time()
                pbar.write(f"Running verification at step {global_step}...")

                lfw_featL, lfw_featR = getFeature_mfn(model, eval_loaders["LFW"], device, flip=True)
                lfw_accs, lfw_thr = evaluation_10_fold_mfn(
                    lfw_featL, lfw_featR, eval_loaders["LFW"].dataset, method="l2_distance"
                )
                lfw_acc = float(np.mean(lfw_accs) * 100)
                lfw_t = float(np.mean(lfw_thr))
                msg = f"Verification done (LFW): acc={lfw_acc:.2f}%, t={lfw_t:.4f}"
                pbar.write(msg)
                write_test(f"step={global_step} {msg}")

                cfp_featL, cfp_featR = getFeature_mfn(model, eval_loaders["CFP-FP"], device, flip=True)
                cfp_accs, cfp_thr = evaluation_10_fold_mfn(
                    cfp_featL, cfp_featR, eval_loaders["CFP-FP"].dataset, method="l2_distance"
                )
                cfp_acc = float(np.mean(cfp_accs) * 100)
                cfp_t = float(np.mean(cfp_thr))
                msg = f"Verification done (CFP-FP): acc={cfp_acc:.2f}%, t={cfp_t:.4f}"
                pbar.write(msg)
                write_test(f"step={global_step} {msg}")

                agedb_featL, agedb_featR = getFeature_mfn(model, eval_loaders["AgeDB-30"], device, flip=True)
                agedb_accs, agedb_thr = evaluation_10_fold_mfn(
                    agedb_featL, agedb_featR, eval_loaders["AgeDB-30"].dataset, method="l2_distance"
                )
                agedb_acc = float(np.mean(agedb_accs) * 100)
                agedb_t = float(np.mean(agedb_thr))
                msg = f"Verification done (AgeDB-30): acc={agedb_acc:.2f}%, t={agedb_t:.4f}"
                pbar.write(msg)
                write_test(f"step={global_step} {msg}")
                write_test(f"step={global_step} verif_time={time.time() - verif_start:.1f}s")

                pbar.disable = False
                pbar.refresh()

                model.train()
                head.train()

            pbar.set_postfix(
                loss=running_loss / max(1, pbar.n),
                acc=correct / max(1, total),
            )

        train_acc = correct / max(total, 1)
        train_loss = running_loss / max(len(train_loader), 1)

        epoch_time = time.time() - epoch_start
        remaining = (num_epochs - (epoch + 1)) * epoch_time
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining))

        print(
            f"epoch {epoch+1}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.6f} "
            f"epoch_time={epoch_time:.1f}s ETA={eta}"
        )
        write_train(
            f"epoch={epoch+1} train_loss={train_loss:.4f} train_acc={train_acc:.6f} "
            f"epoch_time={epoch_time:.1f}s ETA={eta}"
        )

        scheduler.step()

        state = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
        }
        ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
        torch.save(state, ckpt_path)
        torch.save(state, ckpt_dir / "latest.pt")
        print(f"[CKPT] Saved {ckpt_path} and {ckpt_dir / 'latest.pt'}")
#this is for quantizing the model to int8
    if use_torchao_int8:
        int8_model = load_minicnn(emb_dim=emb_dim)[0].eval()
        int8_model.load_state_dict(model.state_dict(), strict=False)
        quantize_(int8_model, Int8WeightOnlyConfig())
        int8_path = ckpt_dir / "latest_int8.pt"
        torch.save({"model_state": int8_model.state_dict()}, int8_path)
        print(f"[CKPT] Saved {int8_path}")


if __name__ == "__main__":
    main()
