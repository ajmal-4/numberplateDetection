import os
import cv2
import numpy as np
import easyocr
import util
from datetime import datetime

class PlateDetector:
    def __init__(self, model_cfg_path, model_weights_path, class_names_path):
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = [j[:-1] for j in f.readlines() if len(j) > 2]

        # Load model
        self.net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
        self.reader = easyocr.Reader(['en'])

    def detect_and_read_plate(self, image):
        H, W, _ = image.shape

        # Convert image and get detections
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), True)
        self.net.setInput(blob)
        detections = util.get_outputs(self.net)

        # Bboxes, class_ids, confidences
        bboxes = []
        class_ids = []
        scores = []

        for detection in detections:
            bbox = detection[:4]
            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
            bbox_confidence = detection[4]
            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])
            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

        # Apply NMS
        bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

        # OCR
        for bbox in bboxes:
            xc, yc, w, h = bbox
            license_plate = image[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :]
            output = self.reader.readtext(license_plate)
            detected_plate_number = output[0][1] if len(output) > 0 else ""
            return detected_plate_number, bboxes
        return None, []

class PlateManager:
    def __init__(self, detected_file='detected_plates.txt', entry_file='entry.txt'):
        self.detected_file = detected_file
        self.entry_file = entry_file

    def save_detected_plate(self, detected_plate_number):
        with open(self.detected_file, "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp}: {detected_plate_number}\n")

    def save_entry(self, detected_plate_number):
        with open(self.entry_file, "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp}: {detected_plate_number}\n")

    def read_entry_file(self):
        entries = {}
        with open(self.entry_file, "r") as file:
            for line in file:
                parts = line.strip().split(": ")
                if len(parts) == 2:
                    timestamp, plate_number = parts
                    entries[plate_number] = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return entries

    def remove_entry(self, detected_plate_number):
        lines = []
        with open(self.entry_file, "r") as file:
            lines = file.readlines()
        with open(self.entry_file, "w") as file:
            for line in lines:
                if detected_plate_number not in line:
                    file.write(line)

def main():
    # Define paths to the model's weights and class names
    model_cfg_path = os.path.join('model', 'cfg', 'darknet-yolov3.cfg')
    model_weights_path = os.path.join('model', 'weights', 'model.weights')
    class_names_path = os.path.join('model', 'class.names')

    # Initialize detector and manager
    detector = PlateDetector(model_cfg_path, model_weights_path, class_names_path)
    manager = PlateManager()

    # Prompt user to select input source
    input_source = input("Select input source: \n1. Laptop camera\n2. Mobile camera\nEnter your choice (1 or 2): ")

    if input_source == '1':
        camera = cv2.VideoCapture(0)  # 0 for default camera, or provide the camera index if you have multiple cameras
    elif input_source == '2':
        print("Connecting to mobile")
        camera = cv2.VideoCapture('http://192.168.1.11:8080/video')
    else:
        print("Invalid choice. Exiting.")
        return

    cv2.namedWindow("Number Plate Detection")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        cv2.imshow('Press Space to Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Press space to capture an image
            captured_image = frame.copy()
            detected_plate_number, bboxes = detector.detect_and_read_plate(captured_image)

            if detected_plate_number:
                print("Detected plate number:", detected_plate_number)
                manager.save_detected_plate(detected_plate_number)

                entry_data = manager.read_entry_file()
                if detected_plate_number in entry_data:
                    entry_time = entry_data[detected_plate_number]
                    exit_time = datetime.now()
                    time_diff = exit_time - entry_time
                    print(f"Time taken: {time_diff}")
                    manager.remove_entry(detected_plate_number)
                else:
                    manager.save_entry(detected_plate_number)

                # Plot bboxes
                for bbox in bboxes:
                    xc, yc, w, h = bbox
                    captured_image = cv2.rectangle(captured_image, (int(xc - (w / 2)), int(yc - (h / 2))), (int(xc + (w / 2)), int(yc + (h / 2))), (0, 255, 0), 5)

                
                # Resize the captured image for the detected window
                cv2.resize(captured_image, (640, 480))

                cv2.imshow('Number Plate Detection', captured_image)

        elif key == ord('q'):  # Press 'q' to exit the loop
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()