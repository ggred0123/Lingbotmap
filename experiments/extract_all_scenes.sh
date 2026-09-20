#!/bin/bash
# Extract 10 Hz frames + GT-interpolated meta.npz for every MCD sequence, one at
# a time.  Sequential on purpose: the work is a linear read of a 3-21 GB bag and
# two of them just contend for the same disk.
#
#   nohup bash experiments/extract_all_scenes.sh > experiments/logs/extract_all.log 2>&1 &
#
# Notes
#   * rosbags lives in .venv-mcd, not the default interpreter.
#   * Every sequence carries /d455b/color/image_raw, so one topic covers all 18 --
#     but the EXTRINSIC differs by platform: kth/tuhh are the handheld suite
#     (hhs_calib.yaml), ntu is the ATV (atv_calib.yaml).  They are not
#     interchangeable; using the wrong one shifts the camera silently.
#   * --count is pose_rows + margin.  Frames outside the GT span are dropped by
#     the extractor itself, so overshooting is free and undershooting truncates.
#   * A sequence with meta.npz already written is skipped, so this is resumable.

set -u
R=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/streaming3d-self-distill
P=/NHNHOME/WORKSPACE/26msit001_A/bispl_lab/youngmin/.venv-mcd/bin/python
MAN=$R/data/mcd/_integrity/gt_manifest.csv
cd "$R" || exit 1

# Nearest transfer first (same campus as the training scene), then a different
# city, then the other platform -- so the most informative frames land earliest.
SCENES="kth_day_10 kth_night_04 kth_night_05 kth_night_01
        tuhh_day_02 tuhh_day_04 tuhh_night_07 tuhh_night_08 tuhh_night_09
        ntu_day_01 ntu_day_02 ntu_day_10 ntu_night_04 ntu_night_08 ntu_night_13"

total=$(echo $SCENES | wc -w); i=0; ok=0; skip=0; fail=0
echo "[start] $(date '+%F %T')  대상 $total 장면"

for s in $SCENES; do
  i=$((i + 1))
  out=$R/data/mcd/$s/frames_10hz
  bag=$R/data/mcd/$s/${s}_d455b.bag
  gt=$R/data/mcd/$s/gt

  if [ -f "$out/meta.npz" ]; then
    n=$(ls "$out" | wc -l)
    echo "[$i/$total] $s  이미 완료 ($n 파일) — 건너뜀"; skip=$((skip + 1)); continue
  fi
  if [ ! -f "$bag" ] || [ ! -f "$gt/spline.csv" ]; then
    echo "[$i/$total] $s  ★ 입력 없음 (bag=$([ -f "$bag" ] && echo o || echo x) gt=$([ -f "$gt/spline.csv" ] && echo o || echo x)) — 건너뜀"
    fail=$((fail + 1)); continue
  fi

  rows=$(awk -F, -v s="$s" '$1==s{print $2}' "$MAN")
  [ -z "$rows" ] && rows=9000
  count=$((rows + 60))

  sz=$(du -h "$bag" | cut -f1)
  echo "[$i/$total] $s  시작 $(date '+%T')  bag $sz  목표 $count 프레임"
  t0=$(date +%s)
  OMP_NUM_THREADS=8 "$P" experiments/mcd_extract.py \
      --bag "$bag" --topic /d455b/color/image_raw --gt "$gt" \
      --start 0 --count "$count" --stride 3 --out "$out" \
      > "$R/experiments/logs/extract_${s}.log" 2>&1
  rc=$?
  el=$((($(date +%s) - t0) / 60))

  if [ $rc -eq 0 ] && [ -f "$out/meta.npz" ]; then
    ok=$((ok + 1))
    # The extractor sweeps candidate offsets and prints the one it chose; a
    # nonzero pick means frame timestamps and the GT spline disagree, which is
    # exactly the misalignment that stays invisible in ATE and blows up rotation.
    echo "        완료 ${el}분 · $(grep -h '\[gt\]' "$R/experiments/logs/extract_${s}.log" | tail -1)"
    echo "        $(grep -h 'chosen offset' "$R/experiments/logs/extract_${s}.log" | tail -1)"
  else
    fail=$((fail + 1))
    echo "        ★ 실패 (rc=$rc, ${el}분): $(grep -m1 -E 'Error|Traceback|No module|SystemExit' "$R/experiments/logs/extract_${s}.log" 2>/dev/null)"
  fi
done

echo "[done] $(date '+%F %T')  성공 $ok · 건너뜀 $skip · 실패 $fail"
df -h "$R" | tail -1
