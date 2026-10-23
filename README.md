# Day/Night Detection

Classical computer-vision classifier that labels DNIM webcam images as
`day`, `evening`, or `night` — no machine learning, no training data.

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Implementation Steps](#implementation-steps)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Labeling Tools](#labeling-tools)
- [FAQ](#faq)
- [Security Notes](#security-notes)

## Why This Exists

The [DNIM dataset](https://www.kaggle.com/datasets/stevemark/daynight-dataset/code) ships thousands of timelapse
webcam frames across 17 camera sequences, each with a companion
`time_stamp/<id>.txt` file recording the capture time of every frame. That's
enough to build and *grade* a day/night classifier without labeling anything
by hand or training a model: the classifier reads pixels only, and the
timestamps — read afterwards, never during prediction — provide the answer
key for scoring.

This project exists to answer one question cheaply: **how well do simple
brightness/color heuristics do at telling day from night from evening,
compared against a dataset that already tells us when each photo was
actually taken?**

## Features

- Pure classical CV (HSV brightness, saturation, sky-strip color) — no
  model weights, no GPU, no training data.
- Predicts every image in every DNIM sequence in one command.
- Writes predictions into an `output/` tree that mirrors the dataset's
  `time_stamp/` layout, with no images copied.
- Automatically derives ground-truth day/evening/night labels from the
  dataset's timestamps and scores predictions against them.
- Produces an overall accuracy figure, a confusion matrix, per-sequence
  accuracy, and a per-image CSV for debugging misclassifications.

## Prerequisites

- Python 3.8+
- `opencv-python` and `numpy`

Install with:

```bash
pip install -r requirements.txt
```

The dataset itself is not included in this repo. Download it from
[Kaggle](https://www.kaggle.com/datasets/stevemark/daynight-dataset/code),
save the archive as `day_night.zip` in the project root, then extract it:

```bash
unzip day_night.zip -d day_night
```

This must produce `DNIM/Image/` and `DNIM/time_stamp/` under
`day_night/DNIM/` at the project root (already done in this repo):

```
day_night/DNIM/Image/<sequence_id>/*.jpg
day_night/DNIM/time_stamp/<sequence_id>.txt
```

## Implementation Steps

This is how the project was built, in order — useful if you're extending it
or reproducing something similar from scratch:

1. **Extract per-image CV features.** For each image: resize to a fixed
   size, convert to HSV, and compute mean brightness, dark/bright pixel
   ratios, sky-strip brightness, sky blue-vs-red bias, and mean saturation.
2. **Score and threshold.** Combine the features into `day_score` /
   `night_score` tallies with hand-picked thresholds, plus a separate
   `evening` (twilight) condition for the ambiguous mid-brightness,
   warm-sky-tone case. See [How It Works](#how-it-works) for the exact
   rules.
3. **Walk the dataset by sequence.** Discover sequence ids from
   `DNIM/Image/`'s subfolders only (never from `time_stamp/`), so
   prediction structurally cannot see the answer key.
4. **Write mirrored output.** For each sequence, write
   `output/time_stamp/<id>.txt` with one `filename label` line per image,
   preserving the image order from the `Image/` folder.
5. **Derive ground truth from timestamps.** Only now are the original
   `time_stamp/<id>.txt` files read, to pull out each image's adjusted
   `HH MM` capture time and map it to a `day` / `evening` / `night` band
   using fixed clock ranges.
6. **Score.** Join predictions and ground truth by filename within each
   sequence, tally correct/incorrect, and build a confusion matrix and
   per-sequence accuracy breakdown.
7. **Report.** Print the results and persist them to
   `output/score_report.txt` and `output/score_details.csv`.

## Usage

Run everything — prediction and scoring — in a single pass:

```bash
python3 main.py
```

Optional flags if your paths differ from the defaults:

```bash
python3 main.py --dnim-root day_night/DNIM --output-root output
```

| Flag | Default | Meaning |
|---|---|---|
| `--dnim-root` | `day_night/DNIM` | Root folder containing `Image/` and `time_stamp/` |
| `--output-root` | `output` | Where predictions and score reports are written |

### Output layout

```
output/
├── time_stamp/
│   ├── 00000850.txt      # "<filename> <predicted label>" per line
│   ├── 00001323.txt
│   └── ...
├── score_report.txt      # overall accuracy, confusion matrix, per-sequence accuracy
└── score_details.csv     # sequence_id, filename, predicted, actual, correct
```

`output/` never contains images — only text/CSV reports, matching the
dataset's directory shape minus the `Image/` tree.

## How It Works

### Prediction (pixels only)

For each image, six features are computed from a resized (320×240) frame:

| Feature | What it captures |
|---|---|
| `mean_brightness` | Overall HSV value channel mean |
| `dark_pixel_ratio` | Fraction of near-black pixels (V < 50) |
| `bright_pixel_ratio` | Fraction of near-saturated-bright pixels (V > 200) |
| `sky_brightness` | Mean brightness of the top 15% of the frame |
| `sky_blue_minus_red` | Blue-minus-red bias in the sky strip (blue sky vs. warm artificial/sunset light) |
| `mean_saturation` | Overall HSV saturation mean |

These feed a simple point-scoring system (`night_score` vs. `day_score`),
plus a standalone `evening` condition for moderate brightness with warm sky
tones:

- Very dark / very bright thresholds on `mean_brightness` dominate the
  score.
- Dark-pixel and bright-spot ratios add secondary night evidence (e.g.
  streetlights on an otherwise black frame).
- Sky brightness and blue-vs-red bias add or subtract from either score.
- If none of that resolves and brightness sits in the 40–110 range with a
  warm (non-blue) sky, the image is called `evening` outright, overriding
  the day/night tally.

None of this ever touches a filename or a timestamp — it's decided purely
from pixel statistics, per the project's core constraint.

### Scoring (ground truth from timestamps)

The dataset's `time_stamp/<id>.txt` rows look like:

```
20151101_072507.jpg 20151101 00 27
```

The last two columns are the image's own capture time, shifted by a fixed
+1 hour offset (not a day/night label). Ground truth is derived from that
adjusted `HH:MM` using fixed clock bands:

| Band | Range |
|---|---|
| `night` | 19:30–05:29 |
| `evening` | 05:30–06:59 and 18:00–19:29 |
| `day` | 07:00–17:59 |

These bands are constants (`DAWN_START`, `DAY_START`, `DAY_END`,
`DUSK_END`) near the top of `main.py` — adjust them if a particular
sequence's real sunrise/sunset falls outside these generic windows.

Predictions and ground truth are then joined by filename within each
sequence. Filenames present on only one side are excluded from accuracy and
reported as "unmatched" rather than counted wrong.

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**
Install dependencies: `pip install -r requirements.txt`. If you're on a
system where `opencv-python` conflicts with a distro package, try
`pip install opencv-python-headless` instead.

**`No sequences found under day_night/DNIM/Image`**
The dataset zip hasn't been extracted, or `--dnim-root` points somewhere
else. Confirm `day_night/DNIM/Image/<id>/` folders exist.

**Accuracy looks low for one specific sequence**
Check that sequence's rows in `output/score_details.csv` — misclassified
`evening` frames predicted as `night` (or vice versa) are the most common
failure mode, since the clock-band ground truth and the brightness-based
`evening` heuristic don't always agree on where twilight starts/ends for a
given camera and season.

**`could not read image` in a result**
`cv2.imread` returned `None` — the file is likely corrupted, unsupported,
or not actually an image despite its extension. That image is labeled
`unknown` and excluded from scoring (it won't match any ground-truth band).

**Predictions and ground-truth files look identical in name**
`output/time_stamp/<id>.txt` and `day_night/DNIM/time_stamp/<id>.txt` share
a filename but live in different directories and hold different content
(`filename label` vs. `filename date HH MM`). The original dataset file is
never modified — only read.

## Labeling Tools

There's no interactive labeling GUI in this repo, and none is needed for
the current workflow: ground truth is derived automatically from the
dataset's own timestamps (see [How It Works](#how-it-works)), not hand
labeled.

If you want to sanity-check or override that automatic labeling — e.g. you
suspect the clock bands mis-time sunrise/sunset for a specific sequence —
the practical options are:

- **Adjust the bands.** Cheapest fix: tune `DAWN_START` / `DAY_START` /
  `DAY_END` / `DUSK_END` in `main.py` for the season/latitude your
  sequences were captured in, then rerun.
- **Manually spot-check via the CSV.** Open `output/score_details.csv` in
  a spreadsheet, filter to a sequence, and eyeball a handful of `actual`
  values against the corresponding frame in `day_night/DNIM/Image/<id>/`.
- **Use a general-purpose image labeling tool** if you want to build a
  hand-verified ground-truth set instead of relying on timestamps —
  [makesense.ai](https://www.makesense.ai) (browser-based, no install) or
  [LabelImg](https://github.com/HumanSignal/labelImg) both work for
  single-label image classification and can export CSV/JSON you could
  swap in as an alternative ground-truth source.

## FAQ

**Why "evening" instead of "twilight" or "dusk/dawn"?**
Matches the three labels requested for this project. Internally the
brightness condition is the same twilight heuristic either name would
describe.

**Does the classifier ever see the timestamp data?**
No. `predict_sequence()` builds its file list from `DNIM/Image/`'s
subfolders only; `classify_time_of_day()` takes an image path and nothing
else. Ground truth is loaded in a separate step, after predictions are
already written to disk.

**Why not just use the timestamp bands as the "prediction" too — wouldn't
that score 100%?**
That would make the score meaningless — it'd be checking the ground-truth
derivation against itself, not evaluating the image classifier. The whole
point is to see how well pixel-only heuristics approximate the true
capture time.

**Can I run this on a different image folder that isn't DNIM-shaped?**
Not directly — `run()` assumes the `Image/<id>/` + `time_stamp/<id>.txt`
layout so it can score. If you just want predictions on an arbitrary flat
folder of images with no scoring, call `classify_time_of_day()` per file
directly; it has no dependency on the DNIM layout.

**Why does accuracy vary so much by sequence?**
Different cameras have different exposure/gain behavior, artificial
lighting, and framing (how much sky is visible in the top 15% of the
frame), all of which shift the brightness thresholds' effectiveness. This
is expected for a heuristic with fixed, non-per-camera thresholds.

## Security Notes

- **No network access.** Everything runs on local files; nothing is
  uploaded or fetched.
- **Untrusted image files.** `cv2.imread` is used to decode arbitrary
  files under `day_night/DNIM/Image/`. Image codecs (libjpeg, libpng, and
  OpenCV's own decoders) have had memory-safety CVEs historically — don't
  point this script at images from untrusted sources without keeping
  `opencv-python` up to date.
- **Filesystem writes are scoped.** All writes go under `--output-root`
  (default `output/`), created with `os.makedirs`; the script never writes
  inside `day_night/DNIM/` and never deletes anything.
- **No secrets, no credentials.** The project has no config files, API
  keys, or auth tokens to leak.
- **Large dataset archive.** `day_night.zip` (~118 MB) sits at the project
  root. If this project is ever put under version control, exclude the zip
  and the extracted `day_night/` and `output/` trees via `.gitignore`
  rather than committing them.
