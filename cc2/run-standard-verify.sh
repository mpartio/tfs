#!/bin/bash
# Standard verification of new model candidate(s) against the campaign's core
# comparison set on the July 2025 test set.
#
# Usage:
#   bash run-standard-verify.sh <output-tag> <candidate-dir> [<candidate-dir> ...]
#
# Example:
#   bash run-standard-verify.sh mynewmodel \
#       /data/tfs/runs/ED12h/my-new-model-12h
#
# Output:
#   runs/verification/<output-tag>/results/{mae,bias,fss,...}.csv
#   runs/verification/<output-tag>/figures/{stamps0,bias_timeseries,...}.png
#   Plus stdout: Phase-0 composite ranking, multi-case dynamic gate, temporal
#                correlations (state and tendency) per candidate.
#
# The 6-model core set is hardcoded:
#   1. trusting-radar (dynamic-event reference, required for multi-case gate)
#   2. cloudcast-production (advection reference)
#   3. kinetic-creative (production CFM)
#   4. candied-circle (S1 baseline α=1.0)
#   5. candied-circle-aclamp85 (current operational best)
#   6. kinetic-creative-aclamp85 (if present — added when spark inference lands)
#
# All other historical models (α-sweep variants, S2 reduced-arena, mad-pan,
# deterministic-module, MEPS) are documented in the handover and live on
# disk for reproducibility but are NOT routinely re-verified.

set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <output-tag> <candidate-dir> [<candidate-dir> ...]"
  exit 1
fi

OUTPUT_TAG="$1"
shift
CANDIDATES=("$@")

REF_DIR=/data/tfs/runs/ED12h
CORE=(
  "$REF_DIR/trusting-radar-12h"
  "$REF_DIR/cloudcast-production-12h"
  "$REF_DIR/kinetic-creative-k16-eta0p05-sig1p0-12h"
  "$REF_DIR/candied-circle-12h"
  "$REF_DIR/candied-circle-aclamp85-12h"
)
# Add kinetic-creative-aclamp85 if it exists (lands when spark finishes)
if [ -f "$REF_DIR/kinetic-creative-aclamp85-12h/predictions.pt" ]; then
  CORE+=("$REF_DIR/kinetic-creative-aclamp85-12h")
fi

# Combined list: core + candidates
ALL_MODELS=("${CORE[@]}" "${CANDIDATES[@]}")

# Validate candidate dirs exist
for d in "${CANDIDATES[@]}"; do
  if [ ! -f "$d/predictions.pt" ]; then
    echo "ERROR: candidate $d has no predictions.pt"
    exit 1
  fi
done

SAVE_PATH="runs/verification/$OUTPUT_TAG"
mkdir -p "$SAVE_PATH"

echo "================================================================"
echo "Verification run: $OUTPUT_TAG"
echo "Core models: ${#CORE[@]}"
echo "New candidates: ${#CANDIDATES[@]}"
echo "Output: $SAVE_PATH"
echo "================================================================"
echo

# Step 1: verify.py — Phase-0 composite, per-lead metrics, stamps
echo "==== Step 1: verify.py ===="
python3 verify.py \
  --path_name "${ALL_MODELS[@]}" \
  --score mae bias fss highk_power_ratio variance_ratio genesis_lysis \
  --save_path "$SAVE_PATH"
echo

# Step 2: Phase-0 composite (computed externally — not yet wired into verify.py)
echo "==== Step 2: Phase-0 composite (bias-weighted) ===="
python3 - <<PY
import pandas as pd, numpy as np
r = '$SAVE_PATH/results'
mae = pd.read_csv(f'{r}/mae.csv')
bias = pd.read_csv(f'{r}/bias.csv')
fss = pd.read_csv(f'{r}/fss.csv')
gen = pd.read_csv(f'{r}/genesis_lysis.csv')

cat_w = {'Clear':0.20,'Partly cloudy':0.40,'Mostly cloudy':0.25,'Overcast':0.15}
fss['w'] = fss['category'].map(cat_w)
fss_w = fss.groupby(['model','timestep']).apply(
    lambda g: (g['fss']*g['w']).sum()/g['w'].sum(), include_groups=False
).reset_index(name='fss_score')
gen_agg = gen.groupby(['model','timestep'])[['genesis_csi','lysis_csi']].mean().reset_index()
gen_agg['gen_score'] = (gen_agg['genesis_csi'] + gen_agg['lysis_csi']) / 2

rows = []
for m in sorted([x for x in mae['model'].unique() if x != 'eulerian-persistence']):
    sm,sb,sf,sg=[],[],[],[]
    for L in [1,4,8,12]:
        mv = mae[(mae['model']==m)&(mae['timestep']==L)]['mae'].iloc[0]
        bv = abs(bias[(bias['model']==m)&(bias['timestep']==L)]['bias'].iloc[0])
        fv = fss_w[(fss_w['model']==m)&(fss_w['timestep']==L)]['fss_score'].iloc[0]
        gv = gen_agg[(gen_agg['model']==m)&(gen_agg['timestep']=={1:3,4:6,8:9,12:12}[L])]['gen_score'].iloc[0]
        sm.append(np.exp(-(mv/0.20)**2)); sb.append(np.exp(-(bv/0.020)**2)); sf.append(fv); sg.append(gv)
    S = 0.20*np.mean(sm)+0.15*np.mean(sb)+0.35*np.mean(sf)+0.30*np.mean(sg)
    rows.append({'model':m,'S':S,'S_MAE':np.mean(sm),'S_BIAS':np.mean(sb),'S_FSS':np.mean(sf),'S_GEN':np.mean(sg)})

print(pd.DataFrame(rows).sort_values('S',ascending=False).to_string(index=False,float_format='%.4f'))
PY
echo

# Step 3: multi-case dynamic-event gate (per candidate vs trusting-radar)
echo "==== Step 3: Multi-case dynamic-event gate (vs trusting-radar) ===="
for c in "${ALL_MODELS[@]}"; do
  name=$(basename "$c")
  if [[ "$c" == *"trusting-radar"* ]]; then continue; fi  # skip the reference
  rate=$(python3 verif/dynamic_case_gate.py \
    --reference "$REF_DIR/trusting-radar-12h" \
    --candidate "$c" 2>&1 | grep "OVERALL CAPTURE RATE" | awk '{print $4}')
  printf "  %-50s  %s\n" "$name" "$rate"
done
echo

# Step 4: Temporal correlation (state + tendency)
echo "==== Step 4: Temporal correlation ===="
TEMPCORR_LOG="$SAVE_PATH/temporal_correlation.txt"
python3 verif/temporal_correlation.py \
  --candidates "${ALL_MODELS[@]}" \
  --leads 1 4 8 12 \
  --mode both \
  | tee "$TEMPCORR_LOG"
echo

# Step 5: Trajectory-coherence floor gate (Verificationist Round B, 2026-05-28).
# Cloudcast-production wins the composite while its per-pixel tendency
# correlation collapses to ~0 from t+4. The composite alone cannot rank
# AR-vs-direct trajectories fairly without this side-check.
echo "==== Step 5: Trajectory-coherence gate ===="
python3 - "$TEMPCORR_LOG" <<'PY'
import re
import sys

log = open(sys.argv[1]).read()
state_re = re.search(r"State correlation.*?(?=\nTendency correlation|\Z)", log, re.S)
tend_re = re.search(r"Tendency correlation.*", log, re.S)


def parse_table(section: str) -> dict:
    out: dict = {}
    in_table = False
    for line in section.splitlines():
        if re.match(r"^model\s+t\+1", line):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("-"):
            continue
        if not line.strip():
            in_table = False
            continue
        m = re.match(
            r"^(\S.{0,49}?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
            line,
        )
        if m:
            out[m.group(1).strip()] = {
                1: float(m.group(2)),
                4: float(m.group(3)),
                8: float(m.group(4)),
                12: float(m.group(5)),
            }
    return out


state = parse_table(state_re.group(0)) if state_re else {}
tend = parse_table(tend_re.group(0)) if tend_re else {}

models = sorted(set(state) | set(tend))
print(f"{'model':<50}  {'st@12':>8}  {'td@4':>8}  {'td@12':>8}  verdict")
print("-" * 92)
for m in models:
    s12 = state.get(m, {}).get(12, float("nan"))
    t4 = tend.get(m, {}).get(4, float("nan"))
    t12 = tend.get(m, {}).get(12, float("nan"))
    fails = []
    if not (s12 >= 0.32):
        fails.append("state@12<0.32")
    if not (t4 >= 0.04):
        fails.append("tend@4<0.04")
    if not (t12 >= 0.01):
        fails.append("tend@12<0.01")
    verdict = "PASS" if not fails else "FAIL: " + ", ".join(fails)
    print(f"{m:<50}  {s12:>8.4f}  {t4:>8.4f}  {t12:>8.4f}  {verdict}")
print()
print("Floors (Verificationist Round B, 2026-05-28):")
print("  state@t+12 >= 0.32   campaign 'not catastrophically broken'")
print("  tend @t+4  >= 0.04   kinetic-creative's value; cloudcast scores ~0.004")
print("  tend @t+12 >= 0.01   kinetic-creative's value; cloudcast scores ~-0.002")
print("Gate is informational, not blocking — composite + capture-rate still authoritative.")
PY
echo

echo "================================================================"
echo "DONE. Results at: $SAVE_PATH/results/"
echo "                  $SAVE_PATH/figures/"
echo "                  $SAVE_PATH/temporal_correlation.txt"
echo "================================================================"
