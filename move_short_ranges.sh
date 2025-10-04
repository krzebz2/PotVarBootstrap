#!/bin/bash

# Arbeitsverzeichnis definieren
BASE_DIR=~/data/analysis

# Alle Ensemble-Ordner durchsuchen
for dir in "$BASE_DIR"/*/results_*/; do
  echo "[INFO] Prüfe $dir"

  for subdir in final_energies final_energies_clean; do
    src="$dir/$subdir"
    dst="$src/ignored_short_range"

    mkdir -p "$dst"

    for file in "$src"/*.dat; do
      # Datei existiert nicht? Überspringen
      [[ -e "$file" ]] || continue

      # Datei-Endung muss 4 Zahlen enthalten: *_t0_t1_ita_itb.dat
      fname=$(basename "$file")
      parts=(${fname//_/ })
      n=${#parts[@]}

      ita=${parts[$((n - 2))]}
      itb=${parts[$((n - 1))]%.dat}

      # Gültigkeit prüfen
      if [[ "$ita" =~ ^[0-9]+$ && "$itb" =~ ^[0-9]+$ ]]; then
        len=$((itb - ita + 1))
        if (( len < 4 )); then
          echo "[MOVE] $fname  →  $subdir/ignored_short_range/"
          mv "$file" "$dst/"
        fi
      else
        echo "[WARN] Kann ita/itb aus $fname nicht lesen"
      fi
    done
  done
done
