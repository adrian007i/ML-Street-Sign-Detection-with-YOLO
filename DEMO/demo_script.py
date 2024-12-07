from ultralytics import YOLO
import cv2
import time
from tqdm import tqdm  # For the progress bar

# Load the YOLO model (replace 'ml_model_built_on_yolo.pt' with your model's path)
model = YOLO('ml_model_built_on_yolo.pt', verbose=False)

# Start video capture from a video file
video_path = 'demo_vid.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the total frame count for the progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the video frame width, height, and FPS for the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up the VideoWriter to save the output video
output_video_path = 'output_video.mp4'  # Specify the output path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_counter = 0  # Track the current frame number

# Initialize the progress bar
progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    frame_counter += 1

    if not ret:
        print("Reached the end of the video or error reading frame.")
        break

    results = None
    
    # Run YOLO inference every 30 frames
    if frame_counter % 2 == 0:
        results = model(frame, verbose=False)
        if results:
            for result in results:
                boxes = result.boxes.xyxy  # Get the bounding box coordinates
                confidences = result.boxes.conf  # Get the confidence scores for each detection

                for box, confidence in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box[:4])  # Convert to int
                    confidence_percentage = confidence.item() * 100  # Convert confidence to percentage

                    # Draw the rectangle around the detected object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

                    # Display the confidence percentage
                    label = f'{confidence_percentage:.2f}%'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, label, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)
 
                out.write(frame) 

    # Update the progress bar
    progress_bar.update(1)

# Release the progress bar
progress_bar.close()

# Release the video capture and writer objects
cap.release()
out.release()

print(f"Video saved at {output_video_path}")