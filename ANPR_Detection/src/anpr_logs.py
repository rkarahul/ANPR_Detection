import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
from supervision.detection.core import Detections
from supervision.detection.annotate import BoxAnnotator
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.draw.color import ColorPalette
from collections import Counter, deque
import traceback 
# Video path and model initialization
VIDEO_PATH_1 = r"ip-camera8-1921682099-lane-nvr-20241001113000-20241001113300-125261_JJpTthra.wmv"  # Change to live feed if needed
model = YOLO("best_openvino_model",task="detect")
# model.fuse()

# Initialize ByteTrack and LineZone
byte_tracker = ByteTrack()
# Initialize PaddleOCR for reading number plates
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize video frames generator (use video capture for live feed)
cap = cv2.VideoCapture(VIDEO_PATH_1)  # Change to live feed
box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=4, text_thickness=4, text_scale=2)

# Variables for FPS calculation
frame_count = 0
start_time = time.time()

# Dictionary to track the highest confidence result for each ID
tracker_results = {}
logged_ids = {}

# Post-processing functions
def num2char(num):
    map_num2char = {'0': 'D', '1': 'T', '2': 'Z', '3': 'B', '4': 'A', 
                     '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
    return map_num2char.get(num, num)
def char2num(char):
    map_char2num = {
        'A': '4', 'B': '8', 'C': '0', 'D': '0', 'G': '6', 'H': '4', 
        'I': '1', 'L': '4', 'O': '0', 'P': '6', 'Q': '0', 'R': '8', 
        'S': '5', 'T': '1', 'U': '0', 'V': '0', 'Z': '2'
    }
    return map_char2num.get(char, char)
def post_process_first_line(plate_text):
    plate_text = plate_text.upper()
    plate_text = ''.join(e for e in plate_text if e.isalnum())
    if len(plate_text) == 4:
        a, b, c, d = plate_text
        a = char2num(a)
        b = char2num(b)
        c = num2char(c)
        plate_text = a + b + c + d
    elif len(plate_text) == 5:
        a, b, c, d, e = plate_text
        a = char2num(a)
        b = char2num(b)
        c = num2char(c)
        d = num2char(d)
        e = char2num(e)
        plate_text = a + b + c + d + e
    return plate_text
# Function to log the best result for each tracker ID based on the last 5 results
def log_best_result(tracker_id, ocr_text, confidence):
    # Initialize the deque to store the last 5 results if not present
    if tracker_id not in tracker_results:
        tracker_results[tracker_id] = deque(maxlen=5)
    
    # Add the current OCR result to the deque for this tracker
    tracker_results[tracker_id].append({"text": ocr_text, "confidence": confidence})
    
    # If the deque has at least 5 results, proceed to find the best one
    if len(tracker_results[tracker_id]) == 5:
        # Find the result with the highest confidence
        best_result = max(tracker_results[tracker_id], key=lambda x: x['confidence'])
        best_text = best_result['text']
        best_confidence = best_result['confidence']
        
        # Log or override the best result if it's higher than the previously logged result for this tracker ID
        if tracker_id in logged_ids:
            # Compare with the previously logged confidence
            if best_confidence > logged_ids[tracker_id]["best_confidence"]:
                # Update the OCR text and confidence but keep the original timestamp
                logged_ids[tracker_id]["best_text"] = best_text
                logged_ids[tracker_id]["best_confidence"] = best_confidence
        else:
            # First time logging this tracker ID, log it with a timestamp
            logged_ids[tracker_id] = {
                "best_text": best_text,
                "best_confidence": best_confidence,
                "log_time": time.strftime('%Y-%m-%d %H:%M:%S')  # Keep the original time
            }
        # Write the updated best result for this tracker to the log file
        with open("toll_logs.txt", "w") as log_file:
            for log_id, log_data in logged_ids.items():
                log_file.write(f"{log_data['log_time']} - ID-{log_id}: {log_data['best_text']} (Confidence: {log_data['best_confidence']:.2f})\n")
# Example logic in the processing loop
while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        results = model(frame)[0]
        detections = Detections.from_ultralytics(results)
        tracked_detections = byte_tracker.update_with_detections(detections=detections)
        if not tracked_detections:
            continue
        
        # Loop through tracked detections
        for i in range(len(tracked_detections.xyxy)):
            box = tracked_detections.xyxy[i]
            tracker_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            plate_region = frame[y1:y2, x1:x2]
            ocr_results = ocr.ocr(plate_region, cls=True)

            if ocr_results[0] is not None:
                # Process OCR results for the current tracker ID
                for result in ocr_results:
                    for line in result:
                        text, confidence_score = line[1]
                        processed_text = post_process_first_line(text)
                        # Log the best result for this tracker after checking the last 5 results
                        log_best_result(tracker_id, processed_text, confidence_score)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID-{tracker_id}: {processed_text} (Conf: {confidence_score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS and display it
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow("Live Number Plate Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        break

cap.release()
cv2.destroyAllWindows()

