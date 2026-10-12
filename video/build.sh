#!/usr/bin/env bash
# Stitch deck cards + live demo into the Stage I video (1280x720, 30 fps, no audio track needed).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p parts
card() { # card <png> <seconds> <out>
  ffmpeg -v error -y -loop 1 -t "$2" -i "cards/$1" -vf "fps=30,format=yuv420p,fade=t=in:st=0:d=0.4,fade=t=out:st=$(awk "BEGIN{print $2-0.4}"):d=0.4" -c:v libx264 -crf 20 -preset medium "parts/$3"
}
card s01.png 6  a.mp4   # title
card s02.png 10 b.mp4   # problem
card s03.png 9  c.mp4   # solution
demo=$(ls raw/*.webm | head -1)
ffmpeg -v error -y -i "$demo" -vf "fps=30,format=yuv420p,fade=t=in:st=0:d=0.4" -c:v libx264 -crf 20 -preset medium parts/d.mp4
card s06.png 9  e.mp4   # accuracy
card s07.png 9  f.mp4   # latency
card s10.png 9  g.mp4   # stage II
card s12.png 6  h.mp4   # outputs
card s01.png 4  i.mp4   # end on title (repo link)
printf "file 'parts/%s'\n" a.mp4 b.mp4 c.mp4 d.mp4 e.mp4 f.mp4 g.mp4 h.mp4 i.mp4 > list.txt
ffmpeg -v error -y -f concat -safe 0 -i list.txt -c copy -movflags +faststart ClosetDoor_StageI_demo.mp4
rm -rf parts list.txt
ffprobe -v error -show_entries format=duration,size -of default=nw=1 ClosetDoor_StageI_demo.mp4
