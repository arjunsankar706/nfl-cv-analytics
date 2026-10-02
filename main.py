import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open NFL video
video = cv2.VideoCapture("videos/RaidersBroncos.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Create output video
output = cv2.VideoWriter(
    "tracked_players.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # Detect and track people
    results = model.track(
        frame,
        classes=[0],
        persist=True,
        verbose=False
    )

    # Draw detection boxes + tracking IDs
    annotated_frame = results[0].plot()

    # Add frame to output video
    output.write(annotated_frame)

    print(f"Processing frame {frame_number}")

# Clean up
video.release()
output.release()

print("\nDONE!")
print(f"Processed {frame_number} frames")
print("Saved: tracked_players.mp4")