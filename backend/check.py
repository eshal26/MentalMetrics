"""
=============================================================================
diagnose_tcav.py  —  TCAV 0% diagnostic script
=============================================================================
Run this to pinpoint exactly why TCAV scores are 0% on most concepts.

Usage:
    python diagnose_tcav.py \
        --model xceptiontime_mdd_v2_statedict.pt \
        --npz   eeg_preprocessed.npz \
        --cav_bank ./cav_bank \
        --edf  "19-channel/MDD S1 EC.edf"

All arguments have the same defaults as explain_subject.py.
=============================================================================
"""

import os
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import mne

warnings.filterwarnings('ignore')

# ── Config (must match explain_subject.py) ────────────────────────────────
SAMPLING_RATE = 256
SEG_LENGTH    = 5 * SAMPLING_RATE
N_CHANNELS    = 19
N_CLASSES     = 2
MDD_CLASS     = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# HELPERS  (minimal copies from explain_subject.py)
# =============================================================================

def get_layer_module(model, layer_name):
    parts = layer_name.split('.')
    mod   = model
    for p in parts:
        mod = list(mod.children())[int(p)] if p.isdigit() else getattr(mod, p)
    return mod


def load_model(model_path):
    from tsai.models.XceptionTimePlus import XceptionTimePlus
    model = XceptionTimePlus(c_in=N_CHANNELS, c_out=N_CLASSES, nf=32,
                             act=nn.LeakyReLU)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_edf_simple(filepath):
    """Minimal EDF load matching preprocess.py channel logic."""
    raw      = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    ch_names = raw.ch_names
    if len(ch_names) > N_CHANNELS:
        eeg_chs = [ch for ch in ch_names if 'EEG' in ch.upper()]
        raw.pick(eeg_chs[:N_CHANNELS] if len(eeg_chs) >= N_CHANNELS
                 else ch_names[:N_CHANNELS])
    eeg = raw.get_data().T.astype('float32')
    # Return first 6 non-overlapping 5-s segments (enough for diagnosis)
    segs = [eeg[s:s + SEG_LENGTH]
            for s in range(0, eeg.shape[0] - SEG_LENGTH + 1, SEG_LENGTH)]
    segs_arr = np.array(segs[:6])                      # (≤6, T, C)
    X        = np.transpose(segs_arr, (0, 2, 1))       # (≤6, C, T)
    return X.astype('float32')


# =============================================================================
# DIAGNOSTIC STEPS
# =============================================================================

def step1_bank_summary(cav_bank_dir):
    print("\n" + "=" * 60)
    print("STEP 1 — CAV bank summary")
    print("=" * 60)

    meta_path = os.path.join(cav_bank_dir, 'cav_bank_meta.json')
    if not os.path.exists(meta_path):
        print(f"  ✗ cav_bank_meta.json not found in {cav_bank_dir}")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  Bank layers : {meta.get('layers')}")
    print(f"  Concepts    : {meta.get('concepts')}")
    print(f"  CAV runs    : {meta.get('n_cav_runs')}")
    print(f"  Pool size   : {meta.get('n_pool')}")

    print(f"\n  Per-file CAV shapes:")
    for fname in sorted(os.listdir(cav_bank_dir)):
        if not fname.endswith('.npz'):
            continue
        data = np.load(os.path.join(cav_bank_dir, fname), allow_pickle=True)
        all_cavs = data['all_cavs']
        accs     = data['cav_accs']
        print(f"    {fname:<40}  all_cavs={all_cavs.shape}  "
              f"acc={accs.mean()*100:.1f}%±{accs.std()*100:.1f}%")

    return meta


def step2_layer_gradient(model, model_path, layer_name):
    print("\n" + "=" * 60)
    print(f"STEP 2 — Gradient sanity at layer: {layer_name}")
    print("=" * 60)

    layer = get_layer_module(model, layer_name)
    mod   = layer
    print(f"  Module type    : {type(mod).__name__}")
    if isinstance(mod, nn.Conv1d):
        print(f"  out_channels   : {mod.out_channels}")
        print(f"  in_channels    : {mod.in_channels}")

    results = []
    for seed in range(3):
        rng   = np.random.default_rng(seed)
        dummy = rng.standard_normal((1, N_CHANNELS, SEG_LENGTH)).astype('float32')
        store = {}

        def hook(m, inp, out, _s=store):
            h = out.detach().requires_grad_(True)
            _s['h'] = h
            return h

        handle = layer.register_forward_hook(hook)
        batch  = torch.from_numpy(dummy).to(device)

        try:
            with torch.enable_grad():
                logits = model(batch)
                score  = logits[0, MDD_CLASS]
                grad_h = torch.autograd.grad(score, store['h'],
                                             retain_graph=False,
                                             allow_unused=False)[0]
        finally:
            handle.remove()

        # Pool over T (FIX-7)
        if grad_h.dim() == 3:
            gp = grad_h[0].mean(dim=-1).detach().cpu().numpy()   # (D,)
        else:
            gp = grad_h[0].detach().cpu().numpy()

        norm = np.linalg.norm(gp)
        results.append(gp)
        print(f"  seed={seed}  grad_pooled shape={gp.shape}  "
              f"norm={norm:.6f}  mean={gp.mean():.6f}  std={gp.std():.6f}")

    diff01 = np.linalg.norm(results[0] - results[1])
    diff02 = np.linalg.norm(results[0] - results[2])
    print(f"\n  Gradient diff seed0 vs seed1: {diff01:.6f}")
    print(f"  Gradient diff seed0 vs seed2: {diff02:.6f}")

    if any(np.linalg.norm(g) < 1e-9 for g in results):
        print("  ✗ ZERO GRADIENTS — wrong layer or graph broken")
    elif diff01 < 1e-9:
        print("  ✗ CONSTANT GRADIENTS — layer insensitive to input")
    else:
        print("  ✓ Gradients look valid")

    return results[0]   # return first gradient for step 3


def step3_dot_product(cav_bank_dir, layer_name, grad_vec):
    print("\n" + "=" * 60)
    print("STEP 3 — CAV · gradient dot products")
    print("=" * 60)

    safe_layer = layer_name.replace('.', '_')

    for concept in ['FAA', 'Theta', 'Alpha_Power', 'Beta_Power', 'TBR', 'Coherence']:
        fpath = os.path.join(cav_bank_dir, f"{concept}_{safe_layer}.npz")
        if not os.path.exists(fpath):
            print(f"  {concept:<15} ✗ file not found: {fpath}")
            continue

        data     = np.load(fpath, allow_pickle=True)
        all_cavs = data['all_cavs']   # (R, D)

        # Dimension check
        if all_cavs.shape[1] != grad_vec.shape[0]:
            print(f"  {concept:<15} ✗ DIM MISMATCH: "
                  f"CAV D={all_cavs.shape[1]} != grad D={grad_vec.shape[0]}")
            continue

        # Dot product of each run's CAV with the gradient
        dots = (all_cavs / (np.linalg.norm(all_cavs, axis=1, keepdims=True) + 1e-10)
                ) @ grad_vec
        pos_frac = float(np.mean(dots > 0))

        print(f"  {concept:<15}  D={all_cavs.shape[1]}  "
              f"dots: mean={dots.mean():+.6f}  std={dots.std():.6f}  "
              f"min={dots.min():+.6f}  max={dots.max():+.6f}  "
              f"pos_frac={pos_frac:.2f}  "
              + ("✓" if 0.1 < pos_frac < 0.9 else "⚠  ALL SAME SIGN"))


def step4_full_dd(model, X, cav_bank_dir, layer_name):
    print("\n" + "=" * 60)
    print("STEP 4 — Full compute_dd_multi_cav on real segments")
    print("=" * 60)

    print(f"  X shape: {X.shape}  (using first {len(X)} segments)")

    safe_layer = layer_name.replace('.', '_')
    layer      = get_layer_module(model, layer_name)

    for concept in ['FAA', 'TBR']:   # spot-check two concepts
        fpath = os.path.join(cav_bank_dir, f"{concept}_{safe_layer}.npz")
        if not os.path.exists(fpath):
            print(f"  {concept}: file not found")
            continue

        all_cavs = np.load(fpath, allow_pickle=True)['all_cavs']   # (R, D)
        R        = all_cavs.shape[0]

        cav_t = torch.tensor(all_cavs, dtype=torch.float32, device=device)
        cav_t = cav_t / (cav_t.norm(dim=1, keepdim=True) + 1e-10)

        X_t    = torch.from_numpy(X).to(device)
        all_dd = []

        for i in range(0, len(X_t), 8):
            batch = X_t[i:i + 8]
            B     = batch.shape[0]
            store = {}

            def fwd_hook(m, inp, out, _s=store):
                h = out.detach().requires_grad_(True)
                _s['h'] = h
                return h

            handle = layer.register_forward_hook(fwd_hook)
            try:
                with torch.enable_grad():
                    logits = model(batch)
                    scores = logits[:, MDD_CLASS]
                    h      = store['h']

                    for b in range(B):
                        grad_h = torch.autograd.grad(
                            outputs=scores[b],
                            inputs=h,
                            retain_graph=(b < B - 1),
                            create_graph=False,
                            allow_unused=False,
                        )[0]

                        if grad_h.dim() == 3:
                            gp = grad_h[b].mean(dim=-1)
                        else:
                            gp = grad_h[b]

                        # Print first segment details
                        if i == 0 and b == 0:
                            gp_np = gp.detach().cpu().numpy()
                            print(f"\n  {concept} seg0:")
                            print(f"    grad_pooled shape : {gp_np.shape}")
                            print(f"    grad_pooled norm  : {np.linalg.norm(gp_np):.6f}")
                            print(f"    grad_pooled std   : {gp_np.std():.6f}")
                            ct = cav_t.cpu().numpy()
                            dots = (ct * gp_np).sum(axis=1)
                            print(f"    dot w/ all CAVs   : mean={dots.mean():+.6f}  "
                                  f"std={dots.std():.6f}  pos={np.mean(dots>0):.2f}")

                        dd_runs = (cav_t * gp.unsqueeze(0)).sum(dim=1)
                        all_dd.append(dd_runs.detach().cpu())
            finally:
                handle.remove()

        dd_mat = torch.stack(all_dd, dim=0).T.numpy()   # (R, N)
        print(f"\n  {concept} full dd_mat ({R} runs × {dd_mat.shape[1]} segs):")
        print(f"    mean={dd_mat.mean():+.6f}  std={dd_mat.std():.6f}")
        print(f"    per-run pos_frac: {(dd_mat > 0).mean(axis=1)}")
        print(f"    TCAV score      : {(dd_mat > 0).mean(axis=1).mean():.4f}")


def step5_cav_direction_check(cav_bank_dir, layer_name, model, X):
    """
    Check if flipping all CAVs would give non-zero TCAV.
    If flipped TCAV is high, the CAV direction is inverted.
    """
    print("\n" + "=" * 60)
    print("STEP 5 — CAV direction flip check")
    print("=" * 60)

    safe_layer = layer_name.replace('.', '_')
    layer      = get_layer_module(model, layer_name)

    for concept in ['FAA', 'Theta', 'TBR']:
        fpath = os.path.join(cav_bank_dir, f"{concept}_{safe_layer}.npz")
        if not os.path.exists(fpath):
            continue

        all_cavs = np.load(fpath, allow_pickle=True)['all_cavs']

        for sign, label in [(+1, 'normal'), (-1, 'flipped')]:
            cavs  = sign * all_cavs
            cav_t = torch.tensor(cavs, dtype=torch.float32, device=device)
            cav_t = cav_t / (cav_t.norm(dim=1, keepdim=True) + 1e-10)

            X_t    = torch.from_numpy(X).to(device)
            all_dd = []

            for i in range(0, len(X_t), 8):
                batch = X_t[i:i + 8]
                B     = batch.shape[0]
                store = {}

                def fwd_hook(m, inp, out, _s=store):
                    h = out.detach().requires_grad_(True)
                    _s['h'] = h
                    return h

                handle = layer.register_forward_hook(fwd_hook)
                try:
                    with torch.enable_grad():
                        logits = model(batch)
                        scores = logits[:, MDD_CLASS]
                        h      = store['h']
                        for b in range(B):
                            grad_h = torch.autograd.grad(
                                outputs=scores[b], inputs=h,
                                retain_graph=(b < B - 1),
                                create_graph=False, allow_unused=False,
                            )[0]
                            gp = grad_h[b].mean(dim=-1) if grad_h.dim() == 3 \
                                 else grad_h[b]
                            dd_runs = (cav_t * gp.unsqueeze(0)).sum(dim=1)
                            all_dd.append(dd_runs.detach().cpu())
                finally:
                    handle.remove()

            dd_mat    = torch.stack(all_dd, dim=0).T.numpy()
            tcav      = float((dd_mat > 0).mean(axis=1).mean())
            print(f"  {concept:<15} {label:<8}  TCAV={tcav*100:.1f}%  "
                  f"mean_dd={dd_mat.mean():+.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCAV 0% diagnostic')
    parser.add_argument('--model',    default='xceptiontime_mdd_v2_statedict.pt')
    parser.add_argument('--npz',      default='eeg_preprocessed.npz')
    parser.add_argument('--cav_bank', default='./cav_bank')
    parser.add_argument('--edf',      default=None,
                        help='Optional EDF file for real-segment tests')
    args = parser.parse_args()

    print("=" * 60)
    print("TCAV DIAGNOSTIC")
    print("=" * 60)
    print(f"  device   : {device}")
    print(f"  model    : {args.model}")
    print(f"  cav_bank : {args.cav_bank}")
    print(f"  edf      : {args.edf or '(none — using random segments)'}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.model)

    # Pick the same layer explain_subject.py uses
    with open(os.path.join(args.cav_bank, 'cav_bank_meta.json')) as f:
        meta    = json.load(f)
    all_layers  = meta.get('layers', [])
    # Last Conv1d with out_channels > 2
    layer_name  = None
    for name in reversed(all_layers):
        try:
            mod = get_layer_module(model, name)
            if isinstance(mod, nn.Conv1d) and mod.out_channels > N_CLASSES:
                layer_name = name
                break
        except Exception:
            pass
    if layer_name is None:
        raise RuntimeError(f"No valid Conv1d layer found in {all_layers}")
    print(f"  Using layer: {layer_name}")

    # Load or generate segments
    if args.edf and os.path.exists(args.edf):
        print(f"\nLoading EDF segments from {args.edf}...")
        X = load_edf_simple(args.edf)
    else:
        print("\nNo EDF provided — using 6 random segments")
        rng = np.random.default_rng(0)
        X   = rng.standard_normal(
                  (6, N_CHANNELS, SEG_LENGTH)).astype('float32')
    print(f"  Segments: {X.shape}")

    # Run steps
    step1_bank_summary(args.cav_bank)
    grad_vec = step2_layer_gradient(model, args.model, layer_name)
    step3_dot_product(args.cav_bank, layer_name, grad_vec)
    step4_full_dd(model, X, args.cav_bank, layer_name)
    step5_cav_direction_check(args.cav_bank, layer_name, model, X)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("""
  How to read results:
  ─────────────────────────────────────────────────────────
  STEP 2  grad norm ≈ 0          → wrong layer, rebuild bank
  STEP 2  diff seed0/seed1 ≈ 0   → constant gradient, wrong layer
  STEP 3  pos_frac = 0.00        → CAV direction is flipped (see step 5)
  STEP 3  pos_frac = 0.50        → random CAV, pool not separable
  STEP 3  dim mismatch           → delete bank and rebuild
  STEP 4  dd std ≈ 0             → gradient is constant per concept
  STEP 5  flipped TCAV >> normal → CAVs need sign correction in bank
  ─────────────────────────────────────────────────────────
  If step 5 shows flipped > normal: add this to explain_subject.py
  in the CAV bank loading section, inside the per-concept loop:

      # Verify CAV points toward MDD using a known positive sample
      pos_centroid = pool_acts[pos_idx].mean(axis=0)
      # Project centroid onto CAV; flip if wrong direction
      for i, cav_vec in enumerate(all_cavs):
          if np.dot(cav_vec, pos_centroid - pool_acts[neg_idx].mean(axis=0)) < 0:
              all_cavs[i] = -all_cavs[i]
    """)