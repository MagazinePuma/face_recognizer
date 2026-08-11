"""
Usage:
    python enroll.py --name "Your Name" --images photo1.jpg photo2.jpg photo3.jpg

Use several images per person (different angles/lighting) for more robust
matching later. Each image should contain exactly one clear face of that person.
"""

import argparse

from pipeline_utils import FaceDetector, FaceEmbedder, load_known_faces, save_known_faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Identity label, e.g. your name")
    parser.add_argument("--images", nargs="+", required=True, help="One or more photos of this person")
    args = parser.parse_args()

    detector = FaceDetector()
    embedder = FaceEmbedder()
    known_faces = load_known_faces()

    embeddings_for_person = known_faces.get(args.name, [])

    for image_path in args.images:
        crops = detector.crop_faces(image_path)
        if not crops:
            print(f"[!] No face detected in {image_path}, skipping.")
            continue
        if len(crops) > 1:
            print(f"[!] Multiple faces detected in {image_path}; using the most confident one.")
            crops.sort(key=lambda c: c[1][4], reverse=True)

        face_crop, box = crops[0]
        embedding = embedder.embed(face_crop)
        embeddings_for_person.append(embedding)
        print(f"[+] Enrolled face from {image_path} (detector confidence {box[4]:.2f})")

    known_faces[args.name] = embeddings_for_person
    save_known_faces(known_faces)
    print(f"\nDone. '{args.name}' now has {len(embeddings_for_person)} reference embedding(s) stored.")


if __name__ == "__main__":
    main()
