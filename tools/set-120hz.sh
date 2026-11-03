#!/bin/sh
# Pone el monitor 3D a 120 Hz (X11 / xrandr), en la resolucion mas alta que admita esa frecuencia.
#   tools/set-120hz.sh                       -> salida y resolucion automaticas (p. ej. 2560x1440 @ 120)
#   tools/set-120hz.sh DP-0                  -> esa salida
#   tools/set-120hz.sh DP-0 120 1920x1080    -> frecuencia y resolucion concretas (100 / 110 / 120 Hz)
OUT="$1"
RATE="${2:-120}"
MODE="$3"
command -v xrandr >/dev/null || { echo "falta xrandr"; exit 1; }
# lista "salida modo" de los modos que tienen esa frecuencia, mayor resolucion primero
CAND=$(xrandr --query | awk -v r="$RATE" -v want="$OUT" -v wm="$MODE" '
  / connected/ { o=$1; next }
  /^ +[0-9]+x[0-9]+/ {
    if (want != "" && o != want) next
    if (wm != "" && $1 != wm) next
    for (i=2;i<=NF;i++) { v=$i; gsub(/[*+]/,"",v); if (v+0 > r-1 && v+0 < r+1) { split($1,a,"x"); print a[1]*a[2], o, $1; break } }
  }' | sort -rn | head -1)
[ -n "$CAND" ] || { echo "No encuentro ningun modo a ${RATE} Hz${MODE:+ en $MODE}${OUT:+ en $OUT}. Mira 'xrandr --query'."; exit 1; }
OUT=$(echo "$CAND" | cut -d' ' -f2)
MODE=$(echo "$CAND" | cut -d' ' -f3)
echo "xrandr --output $OUT --mode $MODE --rate $RATE"
xrandr --output "$OUT" --mode "$MODE" --rate "$RATE"
xrandr --query | grep -A40 "^$OUT " | grep '\*' | head -1
