import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits within each FN/HFOV/surface-count group."
    )
    parser.add_argument("--source", default="data/surf10_12_ul_1104.csv")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--tag", default="20260512")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    return parser.parse_args()


def group_key(row):
    return (round(float(row[0]), 6), round(float(row[1]), 6), int(round(float(row[2]))))


def save_csv(path, data):
    pd.DataFrame(data).to_csv(path, header=None, index=False, encoding="utf-8", float_format="%.12g")


def main():
    args = parse_args()
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    data = np.loadtxt(source, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    groups = {}
    for idx, row in enumerate(data):
        groups.setdefault(group_key(row), []).append(idx)

    rng = np.random.default_rng(args.seed)
    split_indices = {"train": [], "val": [], "test": []}
    summary_rows = []

    for key in sorted(groups):
        indices = np.array(groups[key], dtype=int)
        permuted = rng.permutation(indices)
        n = len(permuted)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(f"Group {key} is too small for the requested ratios: {n} rows")

        train_idx = permuted[:n_train]
        val_idx = permuted[n_train : n_train + n_val]
        test_idx = permuted[n_train + n_val :]

        split_indices["train"].extend(train_idx.tolist())
        split_indices["val"].extend(val_idx.tolist())
        split_indices["test"].extend(test_idx.tolist())

        fn, hfov, n_surf = key
        summary_rows.extend(
            [
                {"split": "train", "fn": fn, "hfov": hfov, "n_surf": n_surf, "count": len(train_idx)},
                {"split": "val", "fn": fn, "hfov": hfov, "n_surf": n_surf, "count": len(val_idx)},
                {"split": "test", "fn": fn, "hfov": hfov, "n_surf": n_surf, "count": len(test_idx)},
            ]
        )

    for split in split_indices:
        split_indices[split] = rng.permutation(np.array(split_indices[split], dtype=int))

    train_path = out_dir / f"scan_lens_train_ul_{args.tag}.csv"
    val_path = out_dir / f"scan_lens_val_ul_{args.tag}.csv"
    test_path = out_dir / f"scan_lens_test_ul_{args.tag}.csv"
    summary_path = out_dir / f"scan_lens_split_ul_{args.tag}_summary.csv"

    save_csv(train_path, data[split_indices["train"]])
    save_csv(val_path, data[split_indices["val"]])
    save_csv(test_path, data[split_indices["test"]])
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")

    print(f"source: {source}")
    print(f"seed: {args.seed}")
    print(f"groups: {len(groups)}")
    print(f"train: {len(split_indices['train'])} -> {train_path}")
    print(f"val:   {len(split_indices['val'])} -> {val_path}")
    print(f"test:  {len(split_indices['test'])} -> {test_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
