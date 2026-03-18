import pickle
import face_recognition
from pathlib import Path

known_dir = Path("known_faces")
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
encodings = []
names = []

for img_path in sorted(known_dir.glob("*.*")):
    name = img_path.stem
    img = face_recognition.load_image_file(str(img_path))
    encs = face_recognition.face_encodings(img)
    if not encs:
        print("No face in", img_path.name)
        continue
    encodings.append(encs[0])
    names.append(name)

with open(out_dir / "face_encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print("Saved", len(encodings), "encodings to outputs/face_encodings.pkl")