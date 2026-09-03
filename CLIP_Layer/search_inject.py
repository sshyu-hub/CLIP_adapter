"""
CLIP Layer Injection Search — 注入式搜索（与最终 adapter/MoA 机制一致）。

为什么不用 search_layers.py：
  search_layers.py 搜的是「哪三层特征 *拼接* 后线性可分最强」（concat + L2norm + 逻辑回归），
  但最终 adapter/MoA 是「在第 N 层后 *注入* 残差、只读末层 CLS」。
  拼接读出 ≠ 注入，所以那个搜索名次不能直接搬来定注入点。

本脚本改成正统注入式：
  在候选层后挂 bottleneck Adapter → 冻结 CLIP 前向 → 读最后一层 CLS → 均值池化 → 分类头，
  与 adapter_train.py / train double.py / model.py 完全一致。
  → 这样搜出来的「最优注入点」才能直接指导后面的 adapter / MoA。

三种模式：
  single  逐个扫描 24 层（单层注入点一阶排序，快，默认）
  combos  对给定 3 层组合注入并排序（直接回答 [13,21,23] vs [15,21,23] 谁好）
  topk    论文式「扫描→圈层→组合」：先 single 扫 24 层，取 top-k 层，
          再对这 k 层做 C(k,3) 三层组合（小范围遍历，带断点续跑）

用法：
  python search_inject.py --mode single
  python search_inject.py --mode single --plot            # 跑完自动画「注入层→Test F1」曲线
  python search_inject.py --mode combos --combos "13,21,23" "15,21,23" "11,17,23"
  python search_inject.py --mode combos --combos "13,21,23" "15,21,23" --seeds 42 7 2024
  python search_inject.py --mode topk --top-k 6 --epochs 5 --bottleneck 128 --seeds 42

层索引说明：vision.encoder.layers 是 24 层的 ModuleList（0-indexed），
  注入点 N 表示「第 N 个 block 输出之后」（与 adapter_train.py 的 ADAPTER_LAYERS 口径一致）。
"""
import os
import sys
import json
import random
import argparse
import itertools

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import CLIPModel

# 复用 config 的路径与常量
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_ROOT, CLIP_MODEL_PATH, OPENFACE_DIR,
    EMOTION_TO_IDX, CLIP_MEAN, CLIP_STD,
    NUM_FRAMES, IMAGE_SIZE, NUM_CLASSES,
)

EMOS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd


# ── 数据加载（与 adapter_train.py / search_layers.py 同源）─────────────
def load_data():
    train_csv = os.path.join(DATA_ROOT, "track1_train_disdim.csv")
    test_csv = os.path.join(DATA_ROOT, "track1_test_dis.csv")
    train_df = pd.read_csv(train_csv)[["name", "discrete"]].dropna(subset=["discrete"])
    test_df = pd.read_csv(test_csv)[["name", "discrete"]].dropna(subset=["discrete"])
    valid = lambda n: os.path.exists(os.path.join(OPENFACE_DIR, f"{n}.npy"))
    tr_names = [n for n in train_df["name"] if valid(n)]
    te_names = [n for n in test_df["name"] if valid(n)]
    tr_labels = {n: EMOTION_TO_IDX[train_df[train_df["name"] == n]["discrete"].values[0]] for n in tr_names}
    te_labels = {n: EMOTION_TO_IDX[test_df[test_df["name"] == n]["discrete"].values[0]] for n in te_names}
    return tr_names, tr_labels, te_names, te_labels


class FDataset(Dataset):
    def __init__(self, names, labels):
        self.names, self.labels = names, labels
        self.dir = OPENFACE_DIR

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        frames = np.load(os.path.join(self.dir, f"{name}.npy"))
        total = frames.shape[0]
        step = max(total / NUM_FRAMES, 1)
        indices = [min(int(i * step), total - 1) for i in range(NUM_FRAMES)]
        frames = frames[indices]
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(frames, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        m = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        return (frames - m) / s, self.labels.get(name, -1)


# ── Adapter（与 adapter_train.py 同构，容量可配）─────────────────────
class Adapter(nn.Module):
    def __init__(self, dim=1024, bottleneck=128):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ── 构建注入式 CLIP（在 layers 后挂 adapter）─────────────────────────
def build_injected_clip(layers, bottleneck):
    clip = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True).to(device)
    vision = clip.vision_model
    for p in vision.parameters():
        p.requires_grad = False
    vision.eval()

    adapters = nn.ModuleDict({str(l): Adapter(bottleneck=bottleneck) for l in layers}).to(device)
    for l in layers:
        a = adapters[str(l)]
        vision.encoder.layers[l].register_forward_hook(lambda m, i, o, a=a: (o[0] + a(o[0]),) + o[1:])
    return vision, adapters


def encode(vision, frames):
    """frames (B,T,C,H,W) -> (B,1024) 末层 CLS 帧间均值（合并 B*T 前向提速）。"""
    B, T, C, H, W = frames.shape
    out = vision(pixel_values=frames.view(B * T, C, H, W))
    cls = out.last_hidden_state[:, 0, :]           # (B*T, 1024)
    return cls.view(B, T, -1).mean(dim=1)          # (B, 1024)


# ── 单候选训练 + 评估 ────────────────────────────────────────────────
def train_and_eval(layers, bottleneck, tr_names, tr_labels, val_names, val_labels,
                   te_names, te_labels, epochs, lr, wd, batch_size, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    vision, adapters = build_injected_clip(layers, bottleneck)
    mlp = MLP().to(device)
    params = list(adapters.parameters()) + list(mlp.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    train_loader = DataLoader(FDataset(tr_names, tr_labels), batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(FDataset(val_names, val_labels), batch_size=batch_size * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

    @torch.no_grad()
    def evaluate(loader):
        preds, gts = [], []
        for frames, labels in loader:
            preds.extend(mlp(encode(vision, frames.to(device))).argmax(-1).cpu().numpy())
            gts.extend(labels.numpy())
        return accuracy_score(gts, preds), f1_score(gts, preds, average="weighted")

    # 与 adapter_train.py 口径一致：每 epoch 看 val，保留 best-val 权重，最后用它测 test
    best_val_f1, best_state = 0.0, None
    for _ in range(epochs):
        for frames, labels in tqdm(train_loader, desc=f"train {layers}", leave=False):
            frames, labels = frames.to(device), labels.to(device)
            loss = F.cross_entropy(mlp(encode(vision, frames)), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        val_acc, val_f1 = evaluate(val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in list(adapters.state_dict().items()) + list(mlp.state_dict().items())}

    if best_state is not None:
        adapters.load_state_dict({k: v for k, v in best_state.items() if k in adapters.state_dict()}, strict=False)
        mlp.load_state_dict({k: v for k, v in best_state.items() if k in mlp.state_dict()}, strict=False)
    val_acc, val_f1 = evaluate(val_loader)
    # 注入式搜索：读出的是「扰动后的末层 CLS」，与最终架构一致
    test_acc, test_f1 = evaluate(
        DataLoader(FDataset(te_names, te_labels), batch_size=batch_size * 2,
                   shuffle=False, num_workers=2, pin_memory=True))

    del vision, adapters, mlp
    torch.cuda.empty_cache()
    return {"val_acc": val_acc, "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1}


# ── 主流程 ────────────────────────────────────────────────────────────
def _run_candidates(candidates, args, tr_train, tr_train_labels, tr_val, tr_val_labels,
                    te_names, te_labels):
    """对一组注入层配置跑多 seed，返回带 mean/std 的结果列表。"""
    results = []
    for layers in candidates:
        entry = {"layers": layers, "runs": []}
        for seed in args.seeds:
            r = train_and_eval(layers, args.bottleneck, tr_train, tr_train_labels,
                               tr_val, tr_val_labels, te_names, te_labels,
                               args.epochs, args.lr, args.wd, args.batch_size, seed)
            entry["runs"].append(r)
            print(f"{layers}  seed={seed}  Val F1={r['val_f1']:.4f}  Test F1={r['test_f1']:.4f}")
        vals = [r["val_f1"] for r in entry["runs"]]
        tests = [r["test_f1"] for r in entry["runs"]]
        entry["val_f1_mean"] = float(np.mean(vals))
        entry["val_f1_std"] = float(np.std(vals))
        entry["test_f1_mean"] = float(np.mean(tests))
        entry["test_f1_std"] = float(np.std(tests))
        results.append(entry)
    return results


def _print_sorted(results, title="=== 注入点排序（按 val_f1 均值） ==="):
    print("\n" + title)
    for r in results:
        print(f"  {r['layers']}  Val={r['val_f1_mean']:.4f}±{r['val_f1_std']:.4f}  "
              f"Test={r['test_f1_mean']:.4f}±{r['test_f1_std']:.4f}")


def _longest_contiguous(layers_asc, ys, thr):
    """在按层号升序的 ys 序列中，找 >= thr 的最长连续区间。返回 (lo, hi) 层号。"""
    best_lo, best_hi, cur_lo = None, None, None
    for i, y in enumerate(ys):
        if y >= thr:
            if cur_lo is None:
                cur_lo = i
        else:
            if cur_lo is not None and (best_lo is None or (i - cur_lo) > (best_hi - best_lo)):
                best_lo, best_hi = cur_lo, i - 1
            cur_lo = None
    if cur_lo is not None and (best_lo is None or (len(ys) - cur_lo) > (best_hi - best_lo)):
        best_lo, best_hi = cur_lo, len(ys) - 1
    return (layers_asc[best_lo], layers_asc[best_hi]) if best_lo is not None else (None, None)


def _summarize_curve(results, margin=0.01):
    """打印峰值 + 好区间（与峰值差距 ≤ margin 的最长连续层区间）。"""
    layers = [r["layers"][0] for r in results]
    means = [r["test_f1_mean"] for r in results]
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    xs = [layers[i] for i in order]
    ys = [means[i] for i in order]
    peak_i = max(range(len(ys)), key=lambda i: ys[i])
    peak_layer, peak_f1 = xs[peak_i], ys[peak_i]
    lo, hi = _longest_contiguous(xs, ys, peak_f1 - margin)
    print(f"\n=== 单层注入曲线（按 Test F1） ===")
    print(f"  峰值：layer {peak_layer}  Test F1={peak_f1:.4f}")
    if lo is not None:
        print(f"  好区间：layers {lo}–{hi}（与峰值差距 ≤ {margin:.3f} 的最长连续段）")
    return xs, ys, peak_layer, peak_f1, lo, hi


def _plot_curve(results, out_path, margin=0.01):
    """matplotlib 画「注入层 → Test F1」曲线，标峰值 + 好区间。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layers"][0] for r in results]
    means = [r["test_f1_mean"] for r in results]
    stds = [r["test_f1_std"] for r in results]
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    xs = [layers[i] for i in order]
    ys = [means[i] for i in order]
    es = [stds[i] for i in order]
    peak_i = max(range(len(ys)), key=lambda i: ys[i])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(xs, ys, yerr=es, marker="o", markersize=5, capsize=3, linewidth=1.5,
                color="#185FA5", ecolor="#85B7EB", label="Test F1 (mean±std)")
    ax.scatter([xs[peak_i]], [ys[peak_i]], color="#E24B4A", zorder=5, s=70, label="peak")
    ax.annotate(f"peak layer {xs[peak_i]}\nTest F1={ys[peak_i]:.4f}",
                xy=(xs[peak_i], ys[peak_i]),
                xytext=(xs[peak_i] + 1.2, ys[peak_i] - 0.012),
                fontsize=10, color="#E24B4A",
                arrowprops=dict(arrowstyle="->", color="#E24B4A"))

    lo, hi = _longest_contiguous(xs, ys, ys[peak_i] - margin)
    if lo is not None:
        ax.axvspan(lo - 0.4, hi + 0.4, color="#E1F5EE", alpha=0.85, zorder=0)
        ax.text(0.5, 0.02, f"good region: layers {lo}–{hi} (within {margin:.3f} of peak)",
                transform=ax.transAxes, ha="center", fontsize=9, color="#0F6E56")

    ax.set_xlabel("injection layer (0-indexed)")
    ax.set_ylabel("Test F1")
    ax.set_title("single-layer injection scan → Test F1")
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"曲线已保存 -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "combos", "topk"], default="single")
    p.add_argument("--combos", nargs="+", default=["13,21,23", "15,21,23", "11,17,23"],
                   help='3层组合，如 "13,21,23" "15,21,23"（空格分隔）')
    p.add_argument("--top-k", type=int, default=6,
                   help="topk 模式：single 扫描后取前 k 层做 C(k,3) 三层组合")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bottleneck", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--plot", action="store_true",
                   help="single 模式跑完后画「注入层 → Test F1」曲线并标峰值/好区间")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    tr_names, tr_labels, te_names, te_labels = load_data()
    random.seed(42)
    random.shuffle(tr_names)
    split = int(0.9 * len(tr_names))
    tr_train, tr_val = tr_names[:split], tr_names[split:]
    tr_train_labels = {n: tr_labels[n] for n in tr_train}
    tr_val_labels = {n: tr_labels[n] for n in tr_val}
    print(f"Train {len(tr_train)}  Val {len(tr_val)}  Test {len(te_names)}  "
          f"frames={NUM_FRAMES}  bottleneck={args.bottleneck}  epochs={args.epochs}")

    if args.mode == "single":
        candidates = [[l] for l in range(24)]
        out = args.out or os.path.join(os.path.dirname(__file__), "inject_results.json")
        results = _run_candidates(candidates, args, tr_train, tr_train_labels,
                                  tr_val, tr_val_labels, te_names, te_labels)
        results.sort(key=lambda r: r["val_f1_mean"], reverse=True)
        _print_sorted(results)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {out}")
        _summarize_curve(results)
        if args.plot:
            _plot_curve(results, os.path.splitext(out)[0] + "_curve.png")
        return

    if args.mode == "combos":
        candidates = [[int(x) for x in c.split(",")] for c in args.combos]
        out = args.out or os.path.join(os.path.dirname(__file__), "inject_results.json")
        results = _run_candidates(candidates, args, tr_train, tr_train_labels,
                                  tr_val, tr_val_labels, te_names, te_labels)
        results.sort(key=lambda r: r["val_f1_mean"], reverse=True)
        _print_sorted(results, title="=== 三层组合排序（按 val_f1 均值） ===")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {out}")
        return

    # ── topk 模式：single 扫描 → 圈 top-k 层 → 小范围三层组合（带断点续跑）──
    out = args.out or os.path.join(os.path.dirname(__file__), "inject_topk_results.json")
    prev = None
    done_combos = set()
    if os.path.exists(out):
        with open(out) as f:
            prev = json.load(f)
        for e in prev.get("combo_results", []):
            done_combos.add(tuple(sorted(e["layers"])))

    # Step 1：单层扫描（若上次已完整保存 24 层排名则复用，避免重复）
    single_ranks = None
    if prev and len(prev.get("single_ranks", [])) == 24:
        single_ranks = prev["single_ranks"]
        print("复用已保存的 24 层单层扫描结果")
    if single_ranks is None:
        single_entries = _run_candidates([[l] for l in range(24)], args,
                                          tr_train, tr_train_labels, tr_val, tr_val_labels,
                                          te_names, te_labels)
        single_ranks = [{"layer": e["layers"][0],
                         "val_f1_mean": e["val_f1_mean"], "val_f1_std": e["val_f1_std"],
                         "test_f1_mean": e["test_f1_mean"], "test_f1_std": e["test_f1_std"]}
                        for e in single_entries]

    single_sorted = sorted(single_ranks, key=lambda x: x["val_f1_mean"], reverse=True)
    top_layers = [e["layer"] for e in single_sorted[:args.top_k]]
    print(f"\nTop-{args.top_k} 单层注入点（按 val_f1_mean）：{top_layers}")
    for e in single_sorted[:args.top_k]:
        print(f"  layer {e['layer']:2d}  Val={e['val_f1_mean']:.4f}±{e['val_f1_std']:.4f}  "
              f"Test={e['test_f1_mean']:.4f}±{e['test_f1_std']:.4f}")

    # Step 2：C(k,3) 三层组合，跳过已算
    combos = [list(c) for c in itertools.combinations(top_layers, 3)]
    pending = [c for c in combos if tuple(sorted(c)) not in done_combos]
    print(f"\n三层组合总数 C({args.top_k},3)={len(combos)}，"
          f"已算 {len(combos) - len(pending)}，待算 {len(pending)}")

    combo_results = prev.get("combo_results", []) if prev else []
    if pending:
        new_results = _run_candidates(pending, args, tr_train, tr_train_labels,
                                      tr_val, tr_val_labels, te_names, te_labels)
        combo_results.extend(new_results)
    combo_results.sort(key=lambda r: r["val_f1_mean"], reverse=True)

    payload = {"top_k": args.top_k, "single_ranks": single_ranks,
               "combo_results": combo_results}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    _print_sorted(combo_results, title="=== 三层组合排序（按 val_f1 均值） ===")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
