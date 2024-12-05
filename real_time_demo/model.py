from ultralytics import YOLO
import cv2

# Load the YOLO model (replace 'yolov8n.pt' with your model's path if needed)


# Load the image
image_path = "road485.png"  # Replace with your image path
img = cv2.imread(image_path)

# Run inference
# model = YOLO()
# results = model(img)

model = YOLO('ml_model_built_on_yolo.pt')
results = model(img)

# # Print prediction results 
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = box.conf[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    label = results[0].names[cls_id]
    print(f"Detected: {label} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

    # Draw bounding boxes on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with predictions
cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image (optional)
cv2.imwrite("predicted_image.png", img)
