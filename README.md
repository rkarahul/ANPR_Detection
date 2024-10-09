# ANPR_Detection
This project implements a complete ANPR (Automatic Number Plate Recognition) system with the following capabilities:
# ANPR Detection

ANPR (Automatic Number Plate Recognition) Detection is a real-time system designed to detect vehicles and recognize number plates using advanced computer vision techniques. This project is built using YOLO (You Only Look Once) for vehicle detection and PaddleOCR for reading license plate text. The goal is to automatically extract number plate information from a video feed or camera input and log the data into a database for further use, such as in toll booths, parking systems, or traffic monitoring.

## Description

This project implements a complete ANPR (Automatic Number Plate Recognition) system with the following capabilities:

- **Vehicle Detection**: Using YOLO, the system detects vehicles in each frame of the video input.
- **License Plate Recognition**: Once a vehicle is detected, the system uses PaddleOCR to extract and recognize the license plate's characters.
- **Tracking Vehicles**: Each detected vehicle is tracked using ByteTrack, ensuring accurate recognition over multiple frames.
- **Database Logging**: Recognized license plate numbers are logged into a MySQL database along with the date and time for record-keeping.
- **Live Processing**: The system processes video frames in real-time, making it suitable for applications such as toll plazas, parking lots, and highway monitoring.

The system is designed for scalability, allowing for easy integration with live camera feeds or offline video files. By logging recognized number plates, it offers a powerful tool for automating tasks like toll collection, security checks, and vehicle tracking.

## Features

- **Real-Time Processing**: The system works on live video feeds or prerecorded videos.
- **High Accuracy**: By leveraging powerful models like YOLO and PaddleOCR, the system achieves high accuracy in detecting and reading number plates.
- **Multiple Vehicle Tracking**: Vehicles are tracked across frames, ensuring correct identification even when multiple vehicles are present.
- **Database Storage**: The recognized number plate information is saved in a MySQL database for later retrieval and analysis.

## Technologies Used

- Python 3.10.15
- OpenCV (for image processing)
- YOLO (for vehicle detection)
- PaddleOCR (for optical character recognition)
- MySQL (for storing number plate data)
- NumPy (for numerical operations)

## Folder Structure

```plaintext
ANPR_Detection/
│
├── src/                        # Source code
│   ├── __init__.py             # To mark this directory as a package
│   └── anpr_system.py          # Your main ANPR system code
│
├── data/                       # Folder for any datasets or sample data
│   └── [your_data_files]       # Place your data files here
│
├── models/                     # Folder for model files
│   └── best_openvino_model      # Your YOLO model file(s)
│
├── requirements.txt            # Python package requirements
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
