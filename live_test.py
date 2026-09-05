import cv2
import torch
import numpy as np
import json
import mediapipe as mp
from src.model import STGCN

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,min_tracking_confidence=0.5)

# 2. Load Label Map
with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
# Invert the map: {index: "label_name"}
inv_label_map = {int(v): k for k, v in label_map.items()}
num_classes = len(inv_label_map)

# 3. Initialize Model
device = torch.device("cpu") # Live test is usually fine on CPU
model = STGCN(num_classes=num_classes).to(device)

# Load the specific weights (Ensure this filename matches your best model)
try:
    model.load_state_dict(torch.load("stgcn_model.pth", map_location=device))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

# 4. Processing Variables
sequence = []
max_frames = 16  # Must match your training max_frames
threshold = 0.5  # Confidence threshold to show label

cap = cv2.VideoCapture(0)

print("Starting Live Test... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip for mirror effect and convert color
    image = cv2.flip(frame, 1)
    debug_image = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Standardize data to zeros if no hand found
    # (ST-GCN expects shape [num_nodes, coordinates])
    frame_landmarks = np.zeros((21, 2)) 

    if results.multi_hand_landmarks:
        # Take the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw landmarks for visual feedback
        mp.solutions.drawing_utils.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Extract coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y])
        frame_landmarks = np.array(landmarks)

    # Maintain sequence of 16 frames
    sequence.append(frame_landmarks)
    sequence = sequence[-max_frames:]

    if len(sequence) == max_frames:
        # Prepare input: [Batch, Frames, Nodes, Coordinates] -> [1, 16, 21, 2]
        # Convert to tensor. Original shape: [1, 16, 21, 2] -> (Batch, Frames, Nodes, Coords)
        input_data = torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(device)

        # Rearrange to: [1, 2, 16, 21] -> (Batch, Coords, Frames, Nodes)
        input_data = input_data.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            output = model(input_data)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, stage_idx = torch.max(probabilities, dim=0)
            
        if confidence.item() > threshold:
            label = inv_label_map[stage_idx.item()]
            text = f"{label} ({confidence.item()*100:.1f}%)"
            cv2.putText(debug_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('PSL Recognition - Live Test', debug_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()