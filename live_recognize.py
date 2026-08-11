"""
Press 'q' to quit the window.

Requires known_faces.pkl to already exist (run enroll.py first), and
opencv-python installed (pip install opencv-python).
"""

import cv2

from pipeline_utils import FaceDetector, FaceEmbedder, load_known_faces, identify

# Only run recognition every N frames to keep things smooth -- embedding
# every single frame is unnecessary and slower than needed for a live feed.
RECOGNIZE_EVERY_N_FRAMES = 5


def main():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    known_faces = load_known_faces()

    if not known_faces:
        print("[!] No enrolled faces found. Run enroll.py first.")
        return

    # 0 is the default camera on macOS (built-in FaceTime camera).
    # If you have multiple cameras attached, try 1, 2, etc.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open the camera. On macOS, make sure your terminal")
        print("    or IDE has camera permission under System Settings > Privacy")
        print("    & Security > Camera.")
        return

    frame_count = 0
    last_results = []  # cached (box, label, color) between recognition passes

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[!] Failed to read from camera.")
            break

        frame_count += 1

        if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            last_results = []
            for face_crop, (x1, y1, x2, y2, conf) in detector.crop_faces(frame):
                embedding = embedder.embed(face_crop)
                name, dist = identify(embedding, known_faces)
                label = f"{name} ({dist:.2f})" if name else f"Unknown ({dist:.2f})"
                color = (0, 255, 0) if name else (0, 0, 255)  # BGR
                last_results.append(((x1, y1, x2, y2), label, color))

        for (x1, y1, x2, y2), label, color in last_results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
