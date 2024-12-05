from ultralytics import YOLO
import cv2
import time

# Load the YOLO model (replace 'ml_model_built_on_yolo.pt' with your model's path)
model = YOLO('ml_model_built_on_yolo.pt')

# Start video capture from a video file
video_path = 'vid.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize the timer
last_inference_time = time.time()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        print("Reached the end of the video or error reading frame.")
        break

    # Check if 5 seconds have passed since the last inference
    current_time = time.time()
    if current_time - last_inference_time >= 5:
        # Run inference on the frame
        results = model(frame)
        
        # Update the last inference time
        last_inference_time = current_time
        
        # Draw bounding boxes and display the confidence percentage
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

    # Display the frame with detections
    cv2.imshow('YOLO Video Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()