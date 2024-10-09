import cv2
import time
import mysql.connector
from ultralytics import YOLO
from paddleocr import PaddleOCR
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.draw.color import ColorPalette
import traceback

# Class to encapsulate the ANPR system functionality
class ANPRSystem:
    def __init__(self, video_path, yolo_model_path):
        # Initialize video path and YOLO model
        self.video_path = video_path
        self.model = YOLO(yolo_model_path, task="detect")
        self.byte_tracker = ByteTrack()
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.cap = cv2.VideoCapture(video_path)
        self.box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=4, text_thickness=4, text_scale=2)

        # Variables for FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        self.tracker_results = {}
        self.logged_ids = {}
        self.database_name = self.create_daily_database()
        self.db_connection, self.db_cursor = self.connect_to_database(self.database_name)
        self.create_table()

    def connect_to_database(self, database_name):
        """Connect to the MySQL database."""
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="admin",
            database=database_name
        )
        db_cursor = db_connection.cursor()
        return db_connection, db_cursor

    def create_daily_database(self):
        """Create a new database for the current date."""
        current_date = time.strftime('%Y_%m_%d')
        database_name = f"toll_log_{current_date}"

        # Connect to the MySQL server and create a database
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="admin"
        )
        db_cursor = db_connection.cursor()
        db_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")
        db_connection.commit()

        # Close the connection to the MySQL server
        db_cursor.close()
        db_connection.close()

        return database_name

    def create_table(self):
        """Create the toll_log table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS toll_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tracker_id VARCHAR(255),
            ocr_text VARCHAR(255),
            confidence FLOAT,
            log_time DATETIME,
            ocr_image LONGBLOB
        );
        """
        self.db_cursor.execute(query)
        self.db_connection.commit()

    def log_best_result(self, tracker_id, ocr_text, confidence, ocr_image):
        """Log the best OCR result for a given tracker ID."""
        tracker_id = int(tracker_id)
        best_confidence = float(confidence)

        # Prepare SQL query for checking existing records
        query_check = "SELECT confidence FROM toll_log WHERE tracker_id = %s ORDER BY log_time DESC LIMIT 1;"
        self.db_cursor.execute(query_check, (tracker_id,))
        previous_confidence = self.db_cursor.fetchone()

        # Get the current time in the desired format
        log_time = time.strftime('%Y-%m-%d %H:%M:%S')

        if previous_confidence is None:
            # Insert new record if no previous entry exists
            query_insert = """
            INSERT INTO toll_log (tracker_id, ocr_text, confidence, log_time, ocr_image)
            VALUES (%s, %s, %s, %s, %s)
            """
            self.db_cursor.execute(query_insert, (tracker_id, ocr_text, best_confidence, log_time, ocr_image))
            self.db_connection.commit()
        else:
            previous_confidence_value = previous_confidence[0]
            if best_confidence > previous_confidence_value:
                query_update = """
                UPDATE toll_log
                SET ocr_text = %s, confidence = %s, log_time = %s, ocr_image = %s
                WHERE tracker_id = %s
                """
                self.db_cursor.execute(query_update, (ocr_text, best_confidence, log_time, ocr_image, tracker_id))
                self.db_connection.commit()

    def num2char(self, num):
        """Map numbers to characters for specific processing."""
        map_num2char = {'0': 'D', '1': 'T', '2': 'Z', '3': 'B', '4': 'A',
                         '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
        return map_num2char.get(num, num)

    def char2num(self, char):
        """Map characters to numbers for specific processing."""
        map_char2num = {
            'A': '4', 'B': '8', 'C': '0', 'D': '0', 'G': '6', 'H': '4',
            'I': '1', 'L': '4', 'O': '0', 'P': '6', 'Q': '0', 'R': '8',
            'S': '5', 'T': '1', 'U': '0', 'V': '0', 'Z': '2'
        }
        return map_char2num.get(char, char)

    def post_process_first_line(self, plate_text):
        """Process the recognized plate text for consistency."""
        plate_text = plate_text.upper()
        plate_text = ''.join(e for e in plate_text if e.isalnum())
        if len(plate_text) in [4, 5]:  # Adjust based on length
            processed_text = ''.join(self.char2num(c) if i < 2 else self.num2char(c) for i, c in enumerate(plate_text))
            return processed_text
        return plate_text

    def process_frame(self, frame):
        """Process a single video frame for detections and OCR."""
        results = self.model(frame)[0]
        detections = Detections.from_ultralytics(results)
        tracked_detections = self.byte_tracker.update_with_detections(detections=detections)
        
        for i in range(len(tracked_detections.xyxy)):
            box = tracked_detections.xyxy[i]
            tracker_id = tracked_detections.tracker_id[i]
            x1, y1, x2, y2 = map(int, box)
            plate_region = frame[y1:y2, x1:x2]
            ocr_results = self.ocr.ocr(plate_region, cls=True)
            if ocr_results and ocr_results[0] is not None:
                for result in ocr_results:
                    for line in result:
                        text, confidence_score = line[1]
                        processed_text = self.post_process_first_line(text)
                        _, buffer = cv2.imencode('.jpg', plate_region)
                        ocr_image = buffer.tobytes()
                        self.log_best_result(tracker_id, processed_text, confidence_score, ocr_image)
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID-{tracker_id}: {processed_text} (Conf: {confidence_score:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def run(self):
        """Main loop for processing video frames."""
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream. Restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self.process_frame(frame)  # Process the current frame

                # Calculate and print FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1:
                    fps = self.frame_count / elapsed_time
                    self.start_time = time.time()  # Reset start time
                    self.frame_count = 0  # Reset frame count
                else:
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                frame=cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
                # Draw FPS on the frame
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                # Show the processed frame with detections
                cv2.imshow("Frame", frame)
                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error: {str(e)}")
                traceback.print_exc()
        self.cleanup()
    def cleanup(self):
        """Release resources and close database connection."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.db_cursor.close()
        self.db_connection.close()

# Run the ANPR system
if __name__ == "__main__":
    VIDEO_PATH = r"data\plate.mp4"  # Change to live feed if needed
    YOLO_MODEL_PATH = r"models\best_openvino_model"
    anpr_system = ANPRSystem(VIDEO_PATH, YOLO_MODEL_PATH)
    anpr_system.run()
