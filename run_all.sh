#!/bin/bash
# Steps 1 to 11 of README.md, then a check of every shipped file (the regenerated outputs included) against CHECKSUMS.sha256.
set -e
cd "$(dirname "$0")"
export PYTHONPYCACHEPREFIX="$(pwd)/.pycache"
P=${PYTHON:-python3}
mkdir -p paper/mst
for s in rev_structural rev_matched rev_sweep rev_fisher rev_lownoise rev_modulation rev_dispersion twin_dispersion real_data_check test_real_data_backside rev_figures make_mst_numbers; do
  t0=$(date +%s); "$P" code/$s.py > logs_$s.txt 2>&1; echo "$s $(( $(date +%s) - t0 )) s"
done
shasum -a 256 -c CHECKSUMS.sha256
