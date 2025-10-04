#!/bin/bash
set -e
set -u

smearing=2  # manuell setzen
analysis_dir=~/data/analysis

# "l" = loop, "s" = submit
mode="s" 

# 1=aic_potential_t_ranges.py, ...
prog_id=8

# Programmliste
programs=(
  "aic_potential_t_ranges.py"
  "aic_weights_check.py"
  "plot_mean_potential.py"
  "plot_eff_mass.py"
  "plot_ct_values.py"
  "bootstrap_parameters_r_ranges.py"
  "aic_parameters_r_ranges.py"
  "plot_cornell_fit.py"
)

prog="${programs[$((prog_id-1))]}"

for dir in "$analysis_dir"/*_sm${smearing}_rmax*; do
  [[ -d "$dir" ]] || continue

  ens=$(basename "$dir")
  resdir="$dir/results_$ens"

  if [[ ! -d "$resdir" ]] || [[ -z $(ls -A "$resdir") ]]; then
    echo "Skip $ens (empty)"
    continue
  fi

  ensemble=$(echo "$ens" | sed 's/_sm[0-9].*//')
  rmax=$(echo "$ens" | sed -n 's/.*rmax\([0-9]\+\).*/\1/p')

  mkdir -p "$resdir/logs"
  logfile="$resdir/logs/${prog%.py}_${ensemble}_sm${smearing}_rmax${rmax}.log"

  if [[ "$mode" == "l" ]]; then
    echo "Run locally: $prog $ensemble $smearing $rmax"
    python3 "$prog" "$ensemble" "$smearing" "$rmax" >"$logfile" 2>&1
  elif [[ "$mode" == "s" ]]; then
    echo "Submit: $prog $ensemble $smearing $rmax"
    sbatch --job-name=${prog%.py} --output="$logfile" --wrap="python3 $prog $ensemble $smearing $rmax"
  fi
done
