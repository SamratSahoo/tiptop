"""Train the conditional flow-matching stroke blender (temporal U-Net, endpoint-anchored deviation).

Consumes ``extract_stroke_flow`` output (``checkpoints/flow_data/stroke_flow.npz``: x [M,T,7] time-domain
strokes, c [M,S,7] arc-length paths) and fits :class:`~tiptop.networks.flow_timing.TrajFlow` with
conditional flow matching on the endpoint-anchored deviation ``delta = x - tau*e`` (e = the stroke's net
displacement / endpoint), conditioned on the timing-stripped path ``c`` and the endpoint ``e``. Writes
``checkpoints/flow_net.pt`` (state_dict + meta with the data scales) for :class:`FlowModel`.

The CFM loss does NOT go to zero (the regression target has irreducible variance); judge the model by its
SAMPLES (``eval_flow.py``), not the loss value.

Run in the openpi venv::

    openpi/.venv/bin/python -m tiptop.networks.train_flow --epochs 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flow_timing import DEFAULT_CKPT, DOF, S_ARC, T_TIME, TrajFlow, cfm_loss  # noqa: E402

MERGED = Path(__file__).resolve().parents[1] / "checkpoints" / "flow_data" / "stroke_flow.npz"


def train(args):
    if not MERGED.exists():
        raise FileNotFoundError(f"{MERGED} not found. Run extract_stroke_flow first.")
    blob = np.load(MERGED)
    x = blob["x"].astype(np.float32)  # [M,T,7]  start-centered positions over time
    c = blob["c"].astype(np.float32)  # [M,S,7]  start-centered path over arc length
    m, T, dof = x.shape
    e = x[:, -1, :]                                   # [M,7] endpoint = net displacement
    tau = np.linspace(0.0, 1.0, T, dtype=np.float32)[None, :, None]
    delta = x - tau * e[:, None, :]                  # delta[:,0]=delta[:,-1]=0 by construction
    delta_std = float(np.sqrt((delta ** 2).mean())) or 1.0
    c_std = float(np.sqrt((c ** 2).mean())) or 1.0
    e_std = float(np.sqrt((e ** 2).mean())) or 1.0
    d_t = torch.as_tensor(delta / delta_std)
    c_t = torch.as_tensor(c / c_std)
    e_t = torch.as_tensor(e / e_std)
    print(f"Loaded {m} strokes.  delta_std={delta_std:.3f} c_std={c_std:.3f} e_std={e_std:.3f}")

    device = args.device
    net = TrajFlow(dof, args.cond_dim, args.t_dim, args.base_ch).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"TrajFlow U-Net: {n_params/1e6:.2f}M params")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(m, generator=g)
    n_val = max(args.batch_size, int(0.05 * m))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    def batches(idx, shuffle):
        order = idx[torch.randperm(len(idx), generator=g)] if shuffle else idx
        for k in range(0, len(order) - 1, args.batch_size):
            yield order[k : k + args.batch_size]

    best = float("inf")
    if not args.out:
        out_path = DEFAULT_CKPT
    else:
        p = Path(args.out)
        out_path = p if p.is_absolute() else (Path(__file__).resolve().parents[1] / p)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        net.train()
        tr = []
        for bi in batches(tr_i, True):
            loss = cfm_loss(net, d_t[bi].to(device), c_t[bi].to(device), e_t[bi].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tr.append(float(loss))
        net.eval()
        with torch.no_grad():
            va = np.mean([float(cfm_loss(net, d_t[val_i].to(device), c_t[val_i].to(device), e_t[val_i].to(device))) for _ in range(4)])
        tag = ""
        if va < best:
            best = va
            meta = {"t_time": T_TIME, "s_arc": S_ARC, "dof": DOF, "c_std": c_std, "delta_std": delta_std,
                    "e_std": e_std, "cond_dim": args.cond_dim, "t_dim": args.t_dim, "base_ch": args.base_ch}
            torch.save({"state_dict": net.state_dict(), "meta": meta}, out_path)
            tag = "  *saved"
        if epoch % max(1, args.epochs // 25) == 0 or tag:
            print(f"epoch {epoch:4d}  train {np.mean(tr):.5f}  val {va:.5f}{tag}", flush=True)

    print(f"\nBest val CFM {best:.5f}. Checkpoint -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--cond-dim", type=int, default=128)
    ap.add_argument("--t-dim", type=int, default=64)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=None)
    train(ap.parse_args())
