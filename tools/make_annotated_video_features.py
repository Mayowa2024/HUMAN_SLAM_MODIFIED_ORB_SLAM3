#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_kitti_poses(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)
    return np.array(poses)


def pose_positions(poses):
    return poses[:, :3, 3]


def align_se3_umeyama_no_scale(est_xyz, gt_xyz):
    """
    Align estimated positions to ground truth using SE(3): rotation + translation, no scale.
    This is similar in spirit to evo_ape -a for stereo trajectories.
    """
    n = min(len(est_xyz), len(gt_xyz))
    src = est_xyz[:n]
    dst = gt_xyz[:n]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    X = src - src_mean
    Y = dst - dst_mean

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean

    aligned = (R @ est_xyz.T).T + t
    return aligned


def read_events(path):
    tracking_by_frame = {}
    special_by_frame = defaultdict(list)

    if path is None or not Path(path).exists():
        return tracking_by_frame, special_by_frame

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(float(row.get("frame_id", -1)))
            except ValueError:
                continue

            event = row.get("event", "")
            module = row.get("module", "")

            # Per-frame tracking state rows.
            if module == "Tracking" and event.startswith("TRACKING"):
                tracking_by_frame[frame_id] = row
            else:
                special_by_frame[frame_id].append(row)

    return tracking_by_frame, special_by_frame


def read_keyframes(path):
    keyframe_by_frame = defaultdict(list)

    if path is None or not Path(path).exists():
        return keyframe_by_frame

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(float(row.get("frame_id", -1)))
            except ValueError:
                continue

            keyframe_by_frame[frame_id].append(row)

    return keyframe_by_frame
def read_features(path):
    features_by_frame = defaultdict(list)

    if path is None or not Path(path).exists():
        return features_by_frame

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(float(row.get("frame_id", -1)))
                x = float(row.get("x", 0))
                y = float(row.get("y", 0))
                has_mappoint = int(row.get("has_mappoint", 0))
                is_outlier = int(row.get("is_outlier", 0))
            except ValueError:
                continue

            features_by_frame[frame_id].append({
                "x": x,
                "y": y,
                "has_mappoint": has_mappoint,
                "is_outlier": is_outlier,
            })

    return features_by_frame


def draw_features_on_image(img, features):
    for ft in features:
        x = int(round(ft["x"]))
        y = int(round(ft["y"]))

        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
            continue

        if ft["is_outlier"] == 1:
            color = (0, 0, 255)      # red = outlier
            radius = 3
        elif ft["has_mappoint"] == 1:
            color = (0, 255, 0)      # green = tracked map point/inlier
            radius = 2
        else:
            color = (0, 255, 255)    # yellow = other logged feature
            radius = 2

        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)

    return img


def read_times(path):
    times = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                times.append(float(line.strip().split()[0]))
    return times


def list_images(image_dir):
    image_dir = Path(image_dir)
    images = sorted(list(image_dir.glob("*.png")))
    if not images:
        images = sorted(list(image_dir.glob("*.jpg")))
    return images


def resize_with_padding(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def draw_text(img, text, x, y, scale=0.6, color=(255, 255, 255), thickness=1):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel_text(img, lines, x, y, line_h=24, scale=0.6):
    for i, line in enumerate(lines):
        draw_text(img, line, x, y + i * line_h, scale=scale)


def make_traj_transform(gt_xyz, est_xyz, panel_w, panel_h, margin=45):
    # Use x-z plane for KITTI top-down view.
    pts = np.vstack([gt_xyz[:, [0, 2]], est_xyz[:, [0, 2]]])

    min_x, min_z = pts.min(axis=0)
    max_x, max_z = pts.max(axis=0)

    range_x = max(max_x - min_x, 1e-6)
    range_z = max(max_z - min_z, 1e-6)

    scale = min(
        (panel_w - 2 * margin) / range_x,
        (panel_h - 2 * margin) / range_z,
    )

    def to_px(xz):
        x = xz[:, 0]
        z = xz[:, 1]

        px = margin + (x - min_x) * scale
        py = panel_h - margin - (z - min_z) * scale

        return np.vstack([px, py]).T.astype(np.int32)

    return to_px


def draw_polyline(img, pts, color, thickness=2):
    if len(pts) < 2:
        return
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def draw_trajectory_panel(gt_xyz, est_xyz, frame_idx, panel_w, panel_h):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    n = min(len(gt_xyz), len(est_xyz))
    i = min(frame_idx, n - 1)

    to_px = make_traj_transform(gt_xyz[:n], est_xyz[:n], panel_w, panel_h)

    gt_px = to_px(gt_xyz[:n, [0, 2]])
    est_px = to_px(est_xyz[:n, [0, 2]])

    # Full paths, faint.
    draw_polyline(panel, gt_px, (90, 90, 90), 1)
    draw_polyline(panel, est_px, (70, 70, 120), 1)

    # Paths up to current frame.
    draw_polyline(panel, gt_px[:i + 1], (180, 180, 180), 2)
    draw_polyline(panel, est_px[:i + 1], (255, 120, 60), 2)

    # Current positions.
    cv2.circle(panel, tuple(gt_px[i]), 5, (220, 220, 220), -1)
    cv2.circle(panel, tuple(est_px[i]), 5, (255, 120, 60), -1)

    draw_text(panel, "Top-down trajectory: x-z", 20, 30, 0.7)
    draw_text(panel, "GT", 20, 60, 0.6, (220, 220, 220))
    draw_text(panel, "ORB-SLAM3", 80, 60, 0.6, (255, 120, 60))

    return panel


def choose_status_color(event):
    if event == "TRACKING_STABLE":
        return (80, 220, 80)
    if event == "TRACKING_WEAK":
        return (0, 220, 255)
    if event == "LOW_INLIERS":
        return (0, 165, 255)
    if event in ("TRACKING_RECENTLY_LOST", "TRACKING_LOST"):
        return (0, 0, 255)
    return (255, 255, 255)


def build_info_lines(frame_idx, timestamp, tracking_row, special_events, keyframe_rows):
    lines = [
        f"Frame: {frame_idx}",
        f"Time: {timestamp:.4f} s",
    ]

    if tracking_row:
        event = tracking_row.get("event", "N/A")
        state_name = tracking_row.get("state_name", "N/A")
        inliers = tracking_row.get("matches_inliers", "N/A")
        current_kf = tracking_row.get("current_kf", "N/A")
        details = tracking_row.get("details", "")

        lines.extend([
            f"Tracking: {event}",
            f"State: {state_name}",
            f"Inliers: {inliers}",
            f"Reference KF: {current_kf}",
            f"Details: {details}",
        ])
    else:
        lines.append("Tracking: no row for this frame")

    if keyframe_rows:
        for kf in keyframe_rows[:2]:
            lines.append(
                "New KF: "
                f"id={kf.get('keyframe_id', 'N/A')} "
                f"inliers={kf.get('matches_inliers', 'N/A')} "
                f"tracked_mps={kf.get('tracked_mappoints', 'N/A')}"
            )

    if special_events:
        lines.append("Events:")
        for ev in special_events[:5]:
            lines.append(
                f"- {ev.get('module', '')}: {ev.get('event', '')} | "
                f"cur_kf={ev.get('current_kf', 'N/A')} "
                f"match_kf={ev.get('matched_kf', 'N/A')} | "
                f"{ev.get('details', '')}"
            )

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--times", required=True)
    parser.add_argument("--gt", required=True, help="KITTI ground truth pose file")
    parser.add_argument("--est", required=True, help="ORB-SLAM3 CameraTrajectory file")
    parser.add_argument("--events", required=True)
    parser.add_argument("--keyframes", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    images = list_images(args.image_dir)
    times = read_times(args.times)

    gt_poses = load_kitti_poses(args.gt)
    est_poses = load_kitti_poses(args.est)

    gt_xyz = pose_positions(gt_poses)
    est_xyz = pose_positions(est_poses)

    n = min(len(images), len(times), len(gt_xyz), len(est_xyz))

    if args.end_frame > 0:
        n = min(n, args.end_frame + 1)

    gt_xyz = gt_xyz[:n]
    est_xyz = align_se3_umeyama_no_scale(est_xyz[:n], gt_xyz[:n])

    tracking_by_frame, special_by_frame = read_events(args.events)
    keyframe_by_frame = read_keyframes(args.keyframes)
    features_by_frame = read_features(args.features)

    out_w, out_h = 1600, 900
    image_w, image_h = 960, 520
    traj_w, traj_h = 640, 520
    info_w, info_h = 1600, 380

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.out}")

    for frame_idx in range(args.start_frame, n, args.step):
        img = cv2.imread(str(images[frame_idx]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        frame_features = features_by_frame.get(frame_idx, [])
        if frame_features:
            img = draw_features_on_image(img, frame_features)

        image_panel = resize_with_padding(img, image_w, image_h)
        traj_panel = draw_trajectory_panel(gt_xyz, est_xyz, frame_idx, traj_w, traj_h)

        info_panel = np.zeros((info_h, info_w, 3), dtype=np.uint8)
        info_panel[:] = (25, 25, 25)

        tracking_row = tracking_by_frame.get(frame_idx)
        special_events = special_by_frame.get(frame_idx, [])
        keyframe_rows = keyframe_by_frame.get(frame_idx, [])

        lines = build_info_lines(
            frame_idx,
            times[frame_idx],
            tracking_row,
            special_events,
            keyframe_rows,
        )
        if frame_features:
            inlier_count = sum(1 for f in frame_features if f["has_mappoint"] == 1 and f["is_outlier"] == 0)
            outlier_count = sum(1 for f in frame_features if f["is_outlier"] == 1)
            lines.append(f"Feature overlay: inliers={inlier_count}, outliers={outlier_count}")

        status_event = tracking_row.get("event", "") if tracking_row else ""
        status_color = choose_status_color(status_event)

        draw_text(info_panel, "ORB-SLAM3 annotated replay", 20, 35, 0.9, (255, 255, 255), 2)
        draw_text(info_panel, status_event if status_event else "NO_TRACKING_ROW", 20, 75, 0.8, status_color, 2)

        draw_panel_text(info_panel, lines, 20, 115, line_h=26, scale=0.62)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[0:image_h, 0:image_w] = image_panel
        canvas[0:traj_h, image_w:out_w] = traj_panel
        canvas[image_h:out_h, 0:out_w] = info_panel[:out_h - image_h, :]

        writer.write(canvas)

    writer.release()
    print(f"Saved annotated video to: {args.out}")


if __name__ == "__main__":
    main()
