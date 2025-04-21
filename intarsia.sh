#!/bin/zsh

set -ue

# Produce two-part cuts for intarsia. The tool size and blur radius
# have been tested on our "1.6mm" Dremel HSS end mills.

NAME="$1"
SIZE="${SIZE:-40}"
PRECISION="${PRECISION:-0.05}"

./depthmap2gcode.py --depth 2.5 --width "$SIZE" --height "$SIZE" --planes 1 --zspace 2 \
  --cutdepth 3.2 --precision "$PRECISION" --tool 1.4:"$NAME"-background.g \
  --result "$NAME"-background-cut.png \
  "$NAME".png

convert "$NAME"-background-cut.png -flop "$NAME"-background-cut-mirror.png 
convert "$NAME"-background-cut-mirror.png \
  -blur "$((6 * 0.05 / $PRECISION))"x"$((3 * 0.05 / $PRECISION))" \
  -level 1%,2% -colors 2 \
  "$NAME"-background-cut-padded.png

./depthmap2gcode.py --depth 2.5 --width "$SIZE" --height "$SIZE" --planes 1 --zspace 2 \
  --cutdepth 3.2 --precision "$PRECISION" --tool 1.4:"$NAME"-foreground.g \
  --result "$NAME"-foreground-cut.png \
  --inverse "$NAME"-background-cut-padded.png

convert "$NAME"-foreground-cut.png \
  "$NAME"-background-cut-mirror.png \
  -compose multiply -composite "$NAME"-collisions.png

display "$NAME"-collisions.png
