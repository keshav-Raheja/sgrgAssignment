# ...existing code...
import os
import pickle
import json
import cv2
import numpy as np
import face_recognition

VIDEO_PATH = "video.mov"
ENCODINGS_FILE = "outputs/face_encodings.pkl"

OUTPUT_VIDEO = "outputs/output_processed.mov"
OUTPUT_JSON = "outputs/energy_scores.json"

FRAME_SKIP = 5
DISPLAY_SCALE = 0.6
FACE_MATCH_THRESHOLD = 0.45

os.makedirs("outputs", exist_ok=True)

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

out_width = int(width * DISPLAY_SCALE)
out_height = int(height * DISPLAY_SCALE)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_width, out_height))

frame_id = 0
prev_gray = None

energy_scores = {}
unknown_counter = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    if frame_id % FRAME_SKIP != 0:
        continue

    # create small frame for faster face detection
    small_scale = 0.5
    small = cv2.resize(frame, (0, 0), fx=small_scale, fy=small_scale)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray)) / 255.0  # 0..1

    if prev_gray is None:
        optical_flow = 0.0
    else:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        optical_flow = float(np.mean(mag))

    prev_gray = gray

    # prepare display frame
    display_frame = cv2.resize(frame, (out_width, out_height))
    disp_scale = out_width / width

    for encoding, location in zip(face_encodings, face_locations):
        top, right, bottom, left = location  # relative to small
        scale = 1.0 / small_scale
        x1 = int(left * scale)
        y1 = int(top * scale)
        x2 = int(right * scale)
        y2 = int(bottom * scale)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # match against known encodings
        name = None
        if known_encodings:
            dists = face_recognition.face_distance(known_encodings, encoding)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist <= FACE_MATCH_THRESHOLD:
                name = known_names[best_idx]
        if name is None:
            unknown_counter += 1
            name = f"UNKNOWN_{unknown_counter:03d}"

        # compute simple energy components
        brightness_score = brightness * 100.0               # 0..100
        eye_openness_score = 50.0                           # placeholder (no landmark model here)
        movement_score = min(100.0, optical_flow * 20.0)    # heuristic scale

        energy = 0.35 * brightness_score + 0.30 * eye_openness_score + 0.35 * movement_score

        energy_scores.setdefault(name, []).append(energy)

        # draw on display frame (scale to display)
        dx1 = int(x1 * disp_scale)
        dy1 = int(y1 * disp_scale)
        dx2 = int(x2 * disp_scale)
        dy2 = int(y2 * disp_scale)
        cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 200, 0), 2)
        label = f"{name} {int(energy)}"
        cv2.putText(display_frame, label, (dx1, max(12, dy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    writer.write(display_frame)

    if frame_id % 100 == 0:
        print(f"Processed frame {frame_id}, faces so far: {len(energy_scores)}")

cap.release()
writer.release()

final_scores = {}

for person, values in energy_scores.items():
    final_scores[person] = float(np.mean(values))

with open(OUTPUT_JSON, "w") as f:
    json.dump(final_scores, f, indent=4)

print("Processing complete")
print("Video saved to:", OUTPUT_VIDEO)
print("Energy JSON saved to:", OUTPUT_JSON)
# ...existing code...