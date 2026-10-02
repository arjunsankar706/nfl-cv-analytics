import cv2
from ultralytics import YOLO

# Load YOLO
model = YOLO("yolo11n.pt")

# Open NFL video
video = cv2.VideoCapture("videos/RaidersBroncos.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Read one frame
success, frame = video.read()

if not success:
    print("Error: Could not read frame.")
    video.release()
    exit()

# Detect people
results = model(frame, classes=[0])

# Draw YOLO's detections onto the frame
annotated_frame = results[0].plot()

# Save the result as an image
cv2.imwrite("detected_players.jpg", annotated_frame)

# Count detected people
player_count = len(results[0].boxes)

print("Players/people detected:", player_count)
print("Saved: detected_players.jpg")

video.release()