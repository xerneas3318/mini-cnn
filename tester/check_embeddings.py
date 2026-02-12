import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="embeddings.npz")
    parser.add_argument("--emb-dim", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Use all unique pairs instead of random sampling",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    emb = data["embeddings"]

    if emb.ndim != 2 or emb.shape[1] != args.emb_dim:
        raise SystemExit(
            f"Expected embeddings shape (N, {args.emb_dim}), got {emb.shape}"
        )

    n = len(emb)
    if n < 2:
        raise SystemExit("Need at least 2 embeddings to compare.")

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    sims = []
    if args.all_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(float(np.dot(emb[i], emb[j])))
    else:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.pairs):
            i, j = rng.choice(n, size=2, replace=False)
            sims.append(float(np.dot(emb[i], emb[j])))

    sims = np.array(sims)
    print(f"pairs: {len(sims)}")
    print(f"mean sim: {sims.mean():.4f}")
    print(f"min sim:  {sims.min():.4f}")
    print(f"max sim:  {sims.max():.4f}")


if __name__ == "__main__":
    main()
