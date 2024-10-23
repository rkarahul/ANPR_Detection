import cv2
import time
from ultralytics import YOLO
from supervision.detection.core import Detections
from supervision.detection.annotate import BoxAnnotator
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.draw.color import ColorPalette
from collections import deque
import nn
import os
import traceback
from warnings import filterwarnings

filterwarnings("ignore")

class LicensePlateRecognizer:
    def __init__(self, detection_model_path, recognition_model_path):
        self.detection_model = YOLO(detection_model_path, task="detect")
        self.recognition_model = nn.Recognition(recognition_model_path)
        self.byte_tracker = ByteTrack()

    def recognize(self, frame):
        results = self.detection_model(frame)[0]
        detections = Detections.from_ultralytics(results)
        tracked_detections = self.byte_tracker.update_with_detections(detections=detections)
        return tracked_detections

    def extract_plate_region(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def ocr(self, plate_region):
        ocr_results = self.recognition_model([plate_region])
        return ocr_results[0] if ocr_results[0] is not None else [], ocr_results[1]

class VideoProcessor:
    def __init__(self, video_path, detection_model_path, recognition_model_path):
        self.cap = cv2.VideoCapture(video_path)
        self.recognizer = LicensePlateRecognizer(detection_model_path, recognition_model_path)
        self.frame_count = 0
        self.start_time = time.time()
        self.box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=4, text_thickness=4, text_scale=2)

    def post_process_first_line(self, plate_text):
        plate_text = plate_text.upper()
        plate_text = ''.join(e for e in plate_text if e.isalnum())
        if len(plate_text) == 4:
            a, b, c, d = plate_text
            a = self.char2num(a)
            b = self.char2num(b)
            c = self.num2char(c)
            plate_text = a + b + c + d
        elif len(plate_text) == 5:
            a, b, c, d, e = plate_text
            a = self.char2num(a)
            b = self.char2num(b)
            c = self.num2char(c)
            d = self.num2char(d)
            e = self.char2num(e)
            plate_text = a + b + c + d + e
        return plate_text

    def char2num(self, char):
        map_char2num = {
            'A': '4', 'B': '8', 'C': '0', 'D': '0', 'G': '6', 'H': '4', 
            'I': '1', 'L': '4', 'O': '0', 'P': '6', 'Q': '0', 'R': '8', 
            'S': '5', 'T': '1', 'U': '0', 'V': '0', 'Z': '2'
        }
        return map_char2num.get(char, char)

    def num2char(self, num):
        map_num2char = {'0': 'D', '1': 'T', '2': 'Z', '3': 'B', '4': 'A', 
                         '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
        return map_num2char.get(num, num)

    def calculate_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0

    def run(self):
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream. Restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                tracked_detections = self.recognizer.recognize(frame)
                if not tracked_detections:
                    continue

                for i in range(len(tracked_detections.xyxy)):
                    box = tracked_detections.xyxy[i]
                    tracker_id = tracked_detections.tracker_id[i]
                    plate_region, (x1, y1, x2, y2) = self.recognizer.extract_plate_region(frame, box)
                    # plate_region=cv2.resize(plate_region, (48,320))
                    print("plate_region",plate_region.shape)
                    recognized_texts, confidence_scores = self.recognizer.ocr(plate_region)

                    if recognized_texts!=None:
                        for idx, text in enumerate(recognized_texts):
                            avg_confidence = sum(confidence_scores[idx]) / len(confidence_scores[idx]) if confidence_scores[idx] else 0.0
                            processed_text = self.post_process_first_line(text)
                            # You can choose to print or display recognized text here if needed
                            print(f"Tracker ID: {tracker_id}, Recognized Text: {processed_text}, Confidence: {avg_confidence:.2f}")

                    # Draw the bounding box and put the label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID-{tracker_id}: {processed_text} (Conf: {avg_confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate FPS and display it
                fps = self.calculate_fps()
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("Live Number Plate Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error processing frame: {e}")
                traceback.print_exc()
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"plate.mp4"
    detection_model_path = r"best_openvino_model"
    recognition_model_path = r'model_rec.onnx'

    processor = VideoProcessor(video_path, detection_model_path, recognition_model_path)
    processor.run()
