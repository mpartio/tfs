"""
Fine-scale FSS placement guard (verificationist-mandated for de-blur eval, 2026-06-04).

The standard S_FSS (phase1) pools mask sizes [5,9,14,22] px = 25/45/70/110 km and OMITS
the 1 px = 5 km scale. A sharpener (CFM, classification+argmax, etc.) that commits pixels
to the right CATEGORY FRACTION but the wrong LOCATION can raise the 25-110 km FSS while
leaving (or lowering) placement skill. The 5 km (1 px) scale IS placement-sensitive and
exposes that. This tool reports FSS across scales INCLUDING 5 km, per category, at the
phase1 leads {1,4,8,12}, and the candidate-minus-reference delta at 5 km.

Pass rule for a de-blur candidate (vs bland-layer-12h): 5 km FSS for Partly-cloudy at
t+4 AND t+12 must be >= reference (delta >= 0). A rise in pooled S_FSS with a DROP at
5 km = sharp-but-misplaced = spectral-metric game, not forecast improvement. REJECT.

Reads /data/tfs/runs/ED12h/<tag>/{predictions,truth}.pt ([N,L,1,H,W]).
Reuses verif.fss.compute_fss_per_leadtime (the same validated FSS as the gates).
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/partio/src/tfs/cc2")
import verif.fss as F

ROOT = "/data/tfs/runs/ED12h"
THRESH = [(0, 0.0625), (0.0625, 0.5625), (0.5625, 0.9375), (0.9375, 1.01)]
CATS = ["Clear", "Partly", "Mostly", "Overcast"]
# pixel mask sizes; 1 px = 5 km grid. Includes the 5 km + 15 km fine scales the
# standard S_FSS omits, plus the standard 25/45/70/110 km for context.
MASKS_PX = [1, 3, 5, 9, 14, 22]
KM = {1: 5, 3: 15, 5: 25, 9: 45, 14: 70, 22: 110}
LEADS = [1, 4, 8, 12]


def load(tag):
    d = f"{ROOT}/{tag}"
    if not os.path.exists(f"{d}/predictions.pt"):
        return None, None
    p = torch.load(f"{d}/predictions.pt", weights_only=False, map_location="cpu").float()
    t = torch.load(f"{d}/truth.pt", weights_only=False, map_location="cpu").float()
    return p, t


def fss_table(tag):
    """Return [cat, size, lead] FSS at LEADS, for the given tag (None if missing)."""
    p, t = load(tag)
    if p is None:
        return None
    # slice to the phase1 leads; compute_fss_per_leadtime wants [N,L,1,H,W]
    p = p[:, LEADS]
    t = t[:, LEADS]
    masks = torch.tensor(MASKS_PX)
    fss = F.compute_fss_per_leadtime(t, p, THRESH, masks)  # [cat, size, lead]
    return fss.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="ED12h tag of the de-blur candidate")
    ap.add_argument("--reference", default="bland-layer-12h", help="ED12h tag to compare against")
    args = ap.parse_args()

    cand = fss_table(args.candidate)
    ref = fss_table(args.reference)
    if cand is None:
        print(f"no tensors for candidate {args.candidate}"); return
    if ref is None:
        print(f"no tensors for reference {args.reference}"); return

    si = MASKS_PX.index(1)  # 5 km scale index
    ci = CATS.index("Partly")  # de-blur regime
    CAT_W = np.array([0.20, 0.40, 0.25, 0.15])
    pooled_idx = [MASKS_PX.index(x) for x in (5, 9, 14, 22)]
    EPS = 0.002  # below this, a pooled "gain" is noise, not a claim worth scrutinising
    print(f"=== fine-scale FSS placement guard: {args.candidate} vs {args.reference} ===\n")

    # scale curve for the candidate (Partly-cloudy) at t+12 — FSS rises with scale; the
    # question is whether the candidate's gain (if any) survives at the fine 5 km scale.
    li12 = LEADS.index(12)
    print("Candidate Partly-cloudy FSS by scale @ t+12:")
    for j, m in enumerate(MASKS_PX):
        print(f"  {KM[m]:>3} km (={m}px): {cand[ci, j, li12]:.4f}")
    print()

    # The guard only MEANS something where the candidate CLAIMS a pooled improvement.
    # Gaming signature = pooled (25-110km, what S_FSS rewards) UP while 5 km (placement) DOWN.
    print("per-lead:  pooled Δ (25-110km, S_FSS)   5km Partly Δ (placement)   classification")
    misplaced = False
    any_claim = False
    for L in (4, 12):
        li = LEADS.index(L)
        cpool = (cand[:, pooled_idx, li].mean(1) * CAT_W).sum()
        rpool = (ref[:, pooled_idx, li].mean(1) * CAT_W).sum()
        dpool = cpool - rpool
        dfine = cand[ci, si, li] - ref[ci, si, li]
        if dpool > EPS:
            any_claim = True
            cls = "GENUINE (placement held)" if dfine >= 0 else "**SHARP-BUT-MISPLACED**"
            if dfine < 0:
                misplaced = True
        else:
            cls = "no pooled gain to validate (cand not better here)"
        print(f"  t+{L:<2}     {dpool:+.4f}                    {dfine:+.4f}                  {cls}")
    print()

    if misplaced:
        verdict = ("FAIL — at >=1 lead the pooled S_FSS gain is NOT matched at 5 km: "
                   "the sharpening added wrong-place detail (spectral-metric game). REJECT.")
    elif any_claim:
        verdict = "PASS — where pooled S_FSS improved, 5 km placement skill held/improved (genuine de-blur)."
    else:
        verdict = ("NEUTRAL — candidate shows no pooled S_FSS improvement over reference, "
                   "so there is no de-blur gain to validate (not a gaming case; just <= reference).")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
