"""
main.py

Classifies each image in the DNIM dataset as 'day', 'evening', or 'night'
using classical computer vision heuristics (no ML model / no training data
needed), writes the predictions into an output directory that mirrors the
dataset's time_stamp layout, then scores those predictions against ground
truth derived from the dataset's own time_stamp files.

The classifier never reads the time_stamp files -- only image pixels are
used to produce a prediction. time_stamp files are used solely afterwards,
to derive ground-truth labels for scoring.

Usage:
    python3 main.py [--dnim-root day_night/DNIM] [--output-root output]
"""

import argparse
import csv
import os

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Ground-truth time-of-day bands, applied to the (already-adjusted) HH:MM
# columns found in day_night/DNIM/time_stamp/<id>.txt. Expressed in minutes
# since midnight for simple range comparisons.
# ---------------------------------------------------------------------------
DAWN_START = 5 * 60 + 30    # 05:30
DAY_START = 7 * 60          # 07:00
DAY_END = 18 * 60           # 18:00 (exclusive)
DUSK_END = 19 * 60 + 30     # 19:30 (exclusive)


def classify_time_of_day(image_path: str) -> dict:
    """
    Returns a dict with the classification label ('day' / 'evening' /
    'night') and the raw feature values used to reach it (useful for
    debugging / threshold tuning). Uses image pixels only.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"time_of_day": "unknown", "error": "could not read image"}

    # Resize for speed/consistency; aspect ratio doesn't matter for stats
    img = cv2.resize(img, (320, 240))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2].astype(np.float32)  # brightness
    s_channel = hsv[:, :, 1].astype(np.float32)  # saturation

    # --- Feature 1: overall mean brightness (0-255) ---
    mean_brightness = float(np.mean(v_channel))

    # --- Feature 2: proportion of very dark pixels (near-black) ---
    dark_pixel_ratio = float(np.mean(v_channel < 50))

    # --- Feature 3: proportion of very bright pixels (lights/sky/sun) ---
    bright_pixel_ratio = float(np.mean(v_channel > 200))

    # --- Feature 4: sky-strip brightness (top 15% of the frame) ---
    h = v_channel.shape[0]
    sky_strip = v_channel[: int(h * 0.15), :]
    sky_brightness = float(np.mean(sky_strip))

    # --- Feature 5: blue-channel dominance in sky strip ---
    # Daylight skies are typically blue-shifted or bright/white.
    # Night skies are dark; artificial-light night scenes often skew orange.
    b, g, r = cv2.split(img.astype(np.float32))
    sky_b = np.mean(b[: int(h * 0.15), :])
    sky_r = np.mean(r[: int(h * 0.15), :])
    sky_blue_minus_red = float(sky_b - sky_r)

    # --- Feature 6: overall saturation mean ---
    mean_saturation = float(np.mean(s_channel))

    # ---------------- Decision logic ----------------
    night_score = 0
    day_score = 0

    # Brightness is the strongest single signal
    if mean_brightness < 60:
        night_score += 3
    elif mean_brightness > 120:
        day_score += 3

    if dark_pixel_ratio > 0.5:
        night_score += 2
    if bright_pixel_ratio > 0.15 and mean_brightness < 100:
        # small bright spots (lights) on a mostly dark frame = night scene
        night_score += 1

    if sky_brightness < 60:
        night_score += 2
    elif sky_brightness > 130:
        day_score += 2

    if sky_blue_minus_red > 10:
        day_score += 1  # blue sky
    elif sky_blue_minus_red < -5:
        night_score += 1  # warm/orange artificial light or sunset tone

    if mean_saturation < 40 and mean_brightness < 90:
        night_score += 1

    # Evening/twilight: moderate brightness, but warm sky tones (sunrise/sunset)
    is_evening = (
        40 <= mean_brightness <= 110
        and sky_blue_minus_red < -5
        and dark_pixel_ratio < 0.5
    )

    if is_evening:
        label = "evening"
    elif night_score > day_score:
        label = "night"
    else:
        label = "day"

    return {
        "time_of_day": label,
        "mean_brightness": round(mean_brightness, 2),
        "dark_pixel_ratio": round(dark_pixel_ratio, 3),
        "bright_pixel_ratio": round(bright_pixel_ratio, 3),
        "sky_brightness": round(sky_brightness, 2),
        "sky_blue_minus_red": round(sky_blue_minus_red, 2),
        "mean_saturation": round(mean_saturation, 2),
    }


def discover_sequences(dnim_root: str) -> list:
    """Returns sorted sequence ids based on the Image/ subfolders only."""
    image_root = os.path.join(dnim_root, "Image")
    return sorted(
        d for d in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, d))
    )


def predict_sequence(dnim_root: str, output_root: str, seq_id: str) -> dict:
    """
    Classifies every image in Image/<seq_id>/ from pixels only and writes
    the predictions to output_root/time_stamp/<seq_id>.txt as
    "<filename> <label>" lines, mirroring the dataset's time_stamp layout.
    """
    seq_dir = os.path.join(dnim_root, "Image", seq_id)
    files = sorted(
        f for f in os.listdir(seq_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    predictions = {}
    for fname in files:
        result = classify_time_of_day(os.path.join(seq_dir, fname))
        predictions[fname] = result["time_of_day"]

    out_dir = os.path.join(output_root, "time_stamp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for fname in files:
            f.write(f"{fname} {predictions[fname]}\n")

    return predictions


def ground_truth_label(hour: int, minute: int) -> str:
    """Maps an adjusted HH:MM capture time to a day/evening/night band."""
    minutes = hour * 60 + minute
    if DAY_START <= minutes < DAY_END:
        return "day"
    if DAWN_START <= minutes < DAY_START or DAY_END <= minutes < DUSK_END:
        return "evening"
    return "night"


def load_ground_truth(dnim_root: str, seq_id: str) -> dict:
    """Parses time_stamp/<seq_id>.txt into {filename: ground_truth_label}."""
    path = os.path.join(dnim_root, "time_stamp", f"{seq_id}.txt")
    truth = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            fname = parts[0]
            hour, minute = int(parts[-2]), int(parts[-1])
            truth[fname] = ground_truth_label(hour, minute)
    return truth


LABELS = ["day", "evening", "night"]


def score_sequence(predictions: dict, truth: dict) -> tuple:
    """
    Compares predictions to ground truth for one sequence by filename.
    Returns (details, unmatched) where details is a list of
    (filename, predicted, actual, correct) and unmatched is a set of
    filenames present on only one side.
    """
    details = []
    common = sorted(set(predictions) & set(truth))
    unmatched = (set(predictions) | set(truth)) - set(common)

    for fname in common:
        pred = predictions[fname]
        actual = truth[fname]
        details.append((fname, pred, actual, pred == actual))

    return details, unmatched


def run(dnim_root: str, output_root: str) -> None:
    seq_ids = discover_sequences(dnim_root)
    if not seq_ids:
        print(f"No sequences found under {os.path.join(dnim_root, 'Image')}")
        return

    confusion = {a: {p: 0 for p in LABELS} for a in LABELS}
    all_details = []
    per_seq_accuracy = {}
    total_unmatched = 0

    for seq_id in seq_ids:
        print(f"Processing sequence {seq_id}...")
        predictions = predict_sequence(dnim_root, output_root, seq_id)
        truth = load_ground_truth(dnim_root, seq_id)
        details, unmatched = score_sequence(predictions, truth)
        total_unmatched += len(unmatched)

        correct = sum(1 for _, _, _, ok in details if ok)
        per_seq_accuracy[seq_id] = (correct, len(details))

        for fname, pred, actual, ok in details:
            confusion[actual][pred] += 1
            all_details.append((seq_id, fname, pred, actual, ok))

    total = len(all_details)
    total_correct = sum(1 for *_, ok in all_details if ok)
    overall_accuracy = total_correct / total if total else 0.0

    # ---------------- Report ----------------
    lines = []
    lines.append(f"Overall accuracy: {total_correct}/{total} = {overall_accuracy:.2%}")
    if total_unmatched:
        lines.append(f"Unmatched filenames (excluded from scoring): {total_unmatched}")
    lines.append("")
    lines.append("Confusion matrix (rows = actual, cols = predicted):")
    header = "actual\\pred".ljust(12) + "".join(l.ljust(10) for l in LABELS)
    lines.append(header)
    for actual in LABELS:
        row = actual.ljust(12) + "".join(str(confusion[actual][p]).ljust(10) for p in LABELS)
        lines.append(row)
    lines.append("")
    lines.append("Per-sequence accuracy:")
    for seq_id in seq_ids:
        correct, count = per_seq_accuracy.get(seq_id, (0, 0))
        acc = correct / count if count else 0.0
        lines.append(f"  {seq_id}: {correct}/{count} = {acc:.2%}")

    report = "\n".join(lines)
    print()
    print(report)

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "score_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    with open(os.path.join(output_root, "score_details.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "filename", "predicted", "actual", "correct"])
        for seq_id, fname, pred, actual, ok in all_details:
            writer.writerow([seq_id, fname, pred, actual, ok])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnim-root", default=os.path.join("day_night", "DNIM"),
                         help="Path to the DNIM dataset root (contains Image/ and time_stamp/)")
    parser.add_argument("--output-root", default="output",
                         help="Path to write predictions and score reports to")
    args = parser.parse_args()

    run(args.dnim_root, args.output_root)
