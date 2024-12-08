from ultralytics import YOLO
import cv2
from tqdm import tqdm

# Load the YOLO model
model = YOLO('ml_model_built_on_yolo.pt', verbose=False)

# Open video file
video_path = 'demo_vid.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup output video writer
output_video_path = 'output_video_with_detections.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_counter = 0
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")
i = 0
while i < 10:
    i += 1
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    # Run inference every 2nd frame
    if frame_counter % 2 == 0:
        results = model(frame, verbose=False)
        detections_exist = False

        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            confidences = result.boxes.conf  # Confidence scores

            if len(boxes) > 0:
                detections_exist = True  # Mark that detections are found

            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box[:4])  # Convert to integers
                conf_percent = confidence

                # class_id = int(box.data[0][-1])
                # print(model.names[class_id]) 
                print(box.data[0][-1])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw confidence label
                label = f'{conf_percent:.2f}%'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the frame only if detections existc
        if detections_exist:
            out.write(frame)

    progress_bar.update(1)

progress_bar.close()
cap.release()
out.release()

print(f"Video with detections saved at {output_video_path}")