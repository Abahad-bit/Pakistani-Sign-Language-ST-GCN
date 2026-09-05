import os
import cv2
import json
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATASET_DIR = "dataset/PakistanSignLanguageDataset"
OUTPUT_DIR = "processed_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for root, dirs, files in os.walk(DATASET_DIR):

    image_files = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_files) == 0:
        continue

    label = os.path.basename(root)

    all_frames = []

    print(f"Processing: {label}")

    for file in image_files:

        img_path = os.path.join(root, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            keypoints = []

            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            all_frames.append({
                "hand_right_keypoints_2d": keypoints
            })

    if len(all_frames) == 0:
        continue

    save_path = os.path.join(OUTPUT_DIR, f"{label}.json")

    with open(save_path, "w") as f:
        json.dump({"people": all_frames}, f)

    print("Saved:", save_path)

print("DONE")