import cv2  # type: ignore
from ultralytics import YOLO  # type: ignore
import time

# Open webcam feed
cap = cv2.VideoCapture(0)

# Load your YOLO model
model = YOLO("BalonModel.pt")

# Set the resolution of the webcam feed
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    time.sleep(0.5)
    status, frame = cap.read()

    if not status:
        break
        
    # Perform object detection
    results = model(frame, stream=True)
    classNames = ["Balon", "Valvula"]

    for res in results:
        boxes = res.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls = int(box.cls[0])
            label = classNames[cls] if cls < len(classNames) else "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the result frame
    cv2.imshow("Camara Termografica", frame)  # Corrected here

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


