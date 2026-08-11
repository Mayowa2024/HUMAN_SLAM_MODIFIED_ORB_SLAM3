#!/usr/bin/env bash
set -e

ORB_DIR="$HOME/ORB_SLAM3"
KITTI_DIR="$HOME/Documents/slam_experiments/datasets/KITTI/dataset"

# Creates a fresh folder every run so old results do not overlap.
RUN_TAG=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="$ORB_DIR/Examples_old/Stereo/annotated_runs_$RUN_TAG"

mkdir -p "$OUT_DIR"

# Edit this list if you want more/less sequences.
SEQS=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09" "10")

cd "$ORB_DIR/Examples_old/Stereo"

for SEQ in "${SEQS[@]}"; do
    echo "========================================"
    echo "Running KITTI sequence $SEQ"
    echo "Output folder: $OUT_DIR/seq_$SEQ"
    echo "========================================"

    RUN_DIR="$OUT_DIR/seq_$SEQ"
    mkdir -p "$RUN_DIR"

    if [[ "$SEQ" == "00" || "$SEQ" == "01" || "$SEQ" == "02" ]]; then
        YAML="KITTI00-02.yaml"
    elif [[ "$SEQ" == "03" ]]; then
        YAML="KITTI03.yaml"
    else
        YAML="KITTI04-12.yaml"
    fi

    rm -f CameraTrajectory.txt orbslam_events.csv orbslam_keyframes.csv

    ./stereo_kitti_old ../../Vocabulary/ORBvoc.txt "$YAML" \
      "$KITTI_DIR/sequences/$SEQ" \
      2>&1 | tee "$RUN_DIR/kitti${SEQ}_console_log.txt"

    mv CameraTrajectory.txt "$RUN_DIR/CameraTrajectory_${SEQ}.txt"
    mv orbslam_events.csv "$RUN_DIR/orbslam_events_${SEQ}.csv"
    mv orbslam_keyframes.csv "$RUN_DIR/orbslam_keyframes_${SEQ}.csv"

    echo "Generating APE plot for sequence $SEQ"

    MPLBACKEND=Agg evo_ape kitti \
      "$KITTI_DIR/poses/$SEQ.txt" \
      "$RUN_DIR/CameraTrajectory_${SEQ}.txt" \
      -a \
      --save_plot "$RUN_DIR/kitti${SEQ}_ape.pdf" \
      > "$RUN_DIR/kitti${SEQ}_ape_metrics.txt"

    echo "Generating trajectory plot for sequence $SEQ"

    MPLBACKEND=Agg evo_traj kitti \
      "$RUN_DIR/CameraTrajectory_${SEQ}.txt" \
      --ref "$KITTI_DIR/poses/$SEQ.txt" \
      --plot_mode xz \
      --save_plot "$RUN_DIR/kitti${SEQ}_traj_xz.pdf"

    echo "Generating annotated video for sequence $SEQ"

    python3 "$ORB_DIR/tools/make_annotated_video.py" \
      --image_dir "$KITTI_DIR/sequences/$SEQ/image_0" \
      --times "$KITTI_DIR/sequences/$SEQ/times.txt" \
      --gt "$KITTI_DIR/poses/$SEQ.txt" \
      --est "$RUN_DIR/CameraTrajectory_${SEQ}.txt" \
      --events "$RUN_DIR/orbslam_events_${SEQ}.csv" \
      --keyframes "$RUN_DIR/orbslam_keyframes_${SEQ}.csv" \
      --out "$RUN_DIR/kitti${SEQ}_annotated.mp4" \
      --fps 10

    echo "Finished sequence $SEQ"
    echo
done

echo "========================================"
echo "All done."
echo "Results saved in:"
echo "$OUT_DIR"
echo "========================================"
