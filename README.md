
# Car Counter using YOLOv8 

## Overview
This project utilizes the YOLOv8 (You Only Look Once) object detection model for detecting cars in a given video.  The primary objective is to accurately count the number of cars in the input video stream.

In addition to YOLOv8, the project incorporates the SORT (Simple Online and Realtime Tracking) algorithm to predict the position of objects in the next frame. SORT helps maintain consistent tracking identities for detected cars across frames, enhancing the accuracy of the car counting process.

## Prerequisites
* Python 3.x
* YOLOv8 
* PyTorch
* CUDA (optional, for GPU acceleration)
* OpenCV
* Ultralytics
* Other dependencies as specified in the requirements.txt file
## Screenshots

![Frame3](https://github.com/Bytecode-Magnum/Yolov8-Car-Counter/assets/99680514/9bc0f951-7e59-4afc-8cb2-54020ec6742b)
![Frame1](https://github.com/Bytecode-Magnum/Yolov8-Car-Counter/assets/99680514/7b7b60bb-7371-409d-95ae-5f6d4c8354c1)


## Installation
1. Download Pre-trained YOLOv8 Weights:
 * Download the pre-trained YOLOv8 weights from the official Ultralytics website:
    [YOLOv8 Weights](https://github.com/ultralytics/yolov5/releases). Place the downloaded weights file (e.g., `yolov8l.pt`) in the weights directory.
2. **Install Dependencies:**

    ```bash
    pip install -U -r requirements.txt
    ```
3. **Clone SORT Repository:**

    Clone the SORT repository to get the `sort.py` file, which is used for predicting the position of objects in the next frame.

    ```bash
    git clone https://github.com/abewley/sort.git
    ```

    Move the `sort.py` file to the project directory.
4. **Run Car Counter:**

    ```bash
    python main.py --input_video your_input_video.mp4
    ```

    Replace `your_input_video.mp4` with the path to your input video file.

    The output video with car counts will be saved in the `output` directory.
