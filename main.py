import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/RaidersBroncos.mp4"
OUTPUT_PATH = "team_tracking.mp4"

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            crop_x1 = x1 + int(box_width * 0.20)
            crop_x2 = x2 - int(box_width * 0.20)

            crop_y1 = y1 + int(box_height * 0.20)
            crop_y2 = y1 + int(box_height * 0.70)

            player_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if player_crop.size == 0:
                continue

            gray = cv2.cvtColor(
                player_crop,
                cv2.COLOR_BGR2GRAY
            )

            brightness = gray.mean()

            if brightness > 115:
                team = "BRONCOS"
                color = (255, 255, 255)
            else:
                team = "RAIDERS"
                color = (0, 0, 0)

            player_id = "?"

            if box.id is not None:
                player_id = int(box.id.item())

            label = f"ID:{player_id} {team}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 30)),
                (x1 + 190, y1),
                color,
                -1
            )

            text_color = (
                (0, 0, 0)
                if team == "BRONCOS"
                else (255, 255, 255)
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2
            )

    output.write(frame)

    if frame_number % 100 == 0:
        print(f"Processing frame {frame_number}")

video.release()
output.release()

print("DONE!")
print(f"Processed {frame_number} frames")
print(f"Saved: {OUTPUT_PATH}")