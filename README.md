# Face Detection + Recognition Pipeline

Two-stage pipeline:
1. **Detection** — your fine-tuned `uralman/yolo26l-widerface` YOLO model finds face bounding boxes.
2. **Recognition** — `facenet-pytorch` (InceptionResnetV1, VGGFace2 weights) turns each detected face into a 512-d embedding and matches it against enrolled identities by distance.

## Setup

```bash
pip install -r requirements.txt
```

Download the fine-tuned detector weights from
https://huggingface.co/uralman/yolo26l-widerface and place the `.pt` file in
this folder as `yolo26l-widerface.pt` (or update `FACE_DETECTOR_WEIGHTS` in
`pipeline_utils.py` to point at wherever you saved it).

## 1. Enroll known people

```bash
python enroll.py --name "Your Name" --images photo1.jpg photo2.jpg photo3.jpg
```

Use 3-5 varied photos (different lighting/angle/expression) per person for
more reliable matching. Re-run with more `--images` any time to add more
reference photos for the same name; embeddings accumulate in `known_faces.pkl`.

## 2. Run recognition

```bash
python recognize.py --image test.jpg --output annotated.jpg
```

Prints each detected face's bounding box, matched identity (or "Unknown"),
and distance score, and optionally saves an annotated copy of the image.

## 3. Live recognition from your webcam

```bash
python live_recognize.py
```

Opens your Mac's built-in camera (device index `0`), runs the pipeline every
few frames, and draws live labeled boxes. Press `q` to quit.

**macOS camera permissions:** the first time you run this, macOS will prompt
whatever app is running the script (Terminal, iTerm, VS Code, etc.) for
camera access. If the window never shows video, check
System Settings > Privacy & Security > Camera and make sure that app is
allowed.

`RECOGNIZE_EVERY_N_FRAMES` in `live_recognize.py` controls how often
detection+recognition actually runs (every 5th frame by default) — raise it
if the feed feels laggy, lower it if matches feel stale as you move.

## Tuning

- `MATCH_THRESHOLD` in `pipeline_utils.py` controls how strict matching is.
  Lower = fewer false positives but more "Unknown" results; higher = the
  opposite. Test against your own photos and adjust.
- `conf` in `FaceDetector.__init__` controls the detector's confidence
  cutoff for what counts as a face.

## Notes

- This pipeline processes single images. For webcam/video, loop
  `detector.crop_faces()` over frames written to a temp file, or adapt
  `FaceDetector.detect()` to accept an in-memory array instead of a path.
- `uralman/yolo26l-widerface` is a third-party fine-tune, not an official
  Ultralytics release — worth spot-checking its detection quality on your
  own photos before relying on it.
