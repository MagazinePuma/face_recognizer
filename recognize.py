"""
Usage:
    python recognize.py --image path/to/photo.jpg
    python recognize.py --image path/to/photo.jpg --output annotated.jpg

Requires known_faces.pkl to already exist (run enroll.py first).
"""

import argparse

from PIL import ImageDraw

from pipeline_utils import FaceDetector, FaceEmbedder, load_known_faces, identify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Image to run recognition on")
    parser.add_argument("--output", default=None, help="Optional path to save an annotated copy")
    args = parser.parse_args()

    detector = FaceDetector()
    embedder = FaceEmbedder()
    known_faces = load_known_faces()

    if not known_faces:
        print("[!] No enrolled faces found. Run enroll.py first.")
        return

    crops = detector.crop_faces(args.image)
    if not crops:
        print("No faces detected.")
        return

    from PIL import Image
    full_img = Image.open(args.image).convert("RGB")
    draw = ImageDraw.Draw(full_img)

    for face_crop, (x1, y1, x2, y2, conf) in crops:
        embedding = embedder.embed(face_crop)
        name, dist = identify(embedding, known_faces)

        label = f"{name} ({dist:.2f})" if name else f"Unknown ({dist:.2f})"
        print(f"Box ({x1},{y1},{x2},{y2}) conf={conf:.2f} -> {label}")

        draw.rectangle([x1, y1, x2, y2], outline="lime" if name else "red", width=3)
        draw.text((x1, max(0, y1 - 12)), label, fill="lime" if name else "red")

    if args.output:
        full_img.save(args.output)
        print(f"\nAnnotated image saved to {args.output}")


if __name__ == "__main__":
    main()
