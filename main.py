import cv2
from gtts import gTTS
from playsound import playsound
import os
from ultralytics import YOLO

# Function to play audio
def play_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "audio.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure you have downloaded yolov8n.pt from the Ultralytics repository

# Load image
image_path = "C:/Users/MITHILESHNIRMIT/OneDrive/Desktop/YOLO/gods-eye/image.jpg"  # Replace with the correct path
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

# Perform object detection
results = model(img)

# Collect detected objects
detected_objects = {}

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get bounding box coordinates and class
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        label = result.names[int(cls)]

        # Draw bounding box and label on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Collect detected objects
        if label in detected_objects:
            detected_objects[label] += 1
        else:
            detected_objects[label] = 1

# Generate audio feedback
audio_text = ", ".join([f"{count} {label}(s)" for label, count in detected_objects.items()])
play_audio(f"I see {audio_text}")

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
