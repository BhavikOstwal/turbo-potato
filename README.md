# Object Detection and Segmentation Script

This script enables object detection and segmentation on images and videos using YOLO and a segmentation model. It provides functionalities to process images/videos and display the results in real-time.

Demo Video - [Youtube Link](https://youtu.be/K_3e20Aoh2U)

## Features
- **YOLO Object Detection**: Detect objects using a trained YOLO model.
- **Segmentation Model**: Perform image segmentation with a TensorFlow Keras model.
- **Drivable Area Detection**: Identify drivable areas using YOLOPv2.
- **Supports Image & Video Processing**: Apply models to both images and videos.
- **Real-time Processing**: Uses OpenCV to visualize processed frames in real-time.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install opencv-python numpy pillow tensorflow torch ultralytics
```
- Note - Install Tensorflow and Torch according to your Cuda version.

## Usage
Run the script with appropriate arguments to process an image or video.

### Arguments
- `--mode`: Choose between `img` (image) or `video` (video) processing.
- `--source`: Path to the image or video file (or camera index for live feed).
- `--model`: Choose between `yolo`, `seg`, or `drivable`.

Cuda is compulsary, if not avialable, use "--devices cpu" (not recommended too slow for segmentation).

### Running Image Processing
```bash
python main.py --mode img --source path/to/image.jpg --model yolo
```

### Running Video Processing
```bash
python main.py --mode video --source path/to/video.mp4 --model seg
```

### Running Drivable Area Detection
```bash
python main.py --mode video --source path/to/video.mp4 --model drivable
```

## Functions Overview

### `run_inference(frame, model_name)`
- Rotates the input frame.
- Runs inference using either the YOLO model (`yolo_model`) or segmentation model (`seg_model`).
- Returns the processed frame with bounding boxes or segmentation mask.

### `process_video(source, model)`
- Captures frames from a video source.
- Processes each frame using the selected model.
- Displays the output in a real-time window.

### `process_image(source, model)`
- Reads an image from the specified path.
- Runs inference using the chosen model.
- Displays the processed image.

### `detect2(source)`
- Loads YOLOPv2 for drivable area and lane line detection.
- Performs inference using the model.
- Applies non-max suppression and segmentation masks.
- Displays results in a real-time window.

## Notes
- Press `q` to exit the visualization windows.
- Ensure the required models (`seg.h5`, `best.pt`, `drivable.pt`) are available in the working directory.
- If you are seeing a rotated image, comment out line 25 in `final.py`.
- You can install weights from this particular drive link :- [Drive Link](https://drive.google.com/drive/folders/1KonFycHYrUBVg0nwwILY_8xKxf6mv18k?usp=sharing). 
- The weights file have to be in the same directory as of the "final.py".
- Drivable model has not been tested without cuda.

## License
This script is open-source and free to use. Modify it as needed for your projects.

## Author
Developed by Bhavik Ostwal, Lavish Singal, Nishant Nehra, Piyush Roy.
Team - Alt + F4