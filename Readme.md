# Video-based Exercise Standardness Detection Algorithm Documentation

## 1. Project Overview
This project is an exercise standardness detection system based on Movenet. It evaluates the accuracy of exercise movements by real-time pose estimation and similarity matching, providing scoring and statistical feedback.

## 2. Features
- Real-time pose estimation: Uses the Movenet model to detect human skeletal keypoints in video frames and extract crucial information.
- Similarity matching: Compares detected poses with pre-defined reference poses to determine the standardness of movements.
- Exercise type switching: Supports real-time switching between different exercise types, such as squats and pull-ups.
- Repetition counting: Counts the number of completed repetitions for each exercise.
- Scoring display: Shows real-time scores and best scores based on movement evaluation results.

## 3. Implementation Details

### 3.1 Detection Logic
For each frame, the skeletal keypoints are captured and compared with the reference library (base_data) to calculate similarity. Each exercise has two postures, “up” and “down.” Completing a down and up sequence counts as one repetition and is scored accordingly.

### 3.2 Pose Estimation
The Movenet model is used for human pose detection in video frames, extracting key skeletal points.

### 3.3 Similarity Matching
The detected pose vectors are compared with the predefined reference poses using cosine similarity, calculating the distance to evaluate the standardness of the movement.

### 3.4 Exercise Type Switching
Real-time keyboard input allows users to switch between different exercise types, including squats and pull-ups.

### 3.5 Repetition Counting
Based on real-time pose detection and exercise type, the number of repetitions for each exercise is counted.

### 3.6 Scoring Display
Based on similarity matching results and repetition counting, real-time scores and best scores are calculated and displayed in the video.

## 4. Directory Structure

- `base_data/` stores reference pose data.
- `movenet/` contains the main model files.
- `result/` outputs the processed result videos.
- `utils/` holds utility scripts for Movenet pose detection.
- `video/` includes video materials needed for detection.
- `main.py` is the main program for exercise pose recognition.
- `cameraTest.py` is used for camera testing (for testing purposes only).
- `setup.py` is the Python build optimization tool.
- `17pose.png` is a reference image with 17 human keypoints.

## 5. Environment Setup

1. **Environment Installation**: Extract `movenet.tar.gz` from the project into your `miniconda3` (or `conda3`) `/envs` directory.
2. **Open Terminal**: Launch the included `miniconda3 prompt` command line tool.
3. **Activate Environment**: Enter `conda activate movenet`.
4. **Change Directory**: Use the `cd` command to navigate to the project directory.
5. **Select Mode**: At the beginning of the `main` function, choose whether to enable the camera and select the exercise type.
6. **Run Command**: Execute `python main.py` to start the program.

## 6. Instructions

- Run the `main()` function to start the project.
- Switch exercise types with keyboard shortcuts: press `1` for squats and `2` for pull-ups.
- Press `ESC` to exit the program.
- Output videos in `.mp4` format will be saved in the `result/` directory.

## 7. Notes

- This project is intended for single-person exercise scenarios.
- Currently, only two exercise types are supported: squats and pull-ups.
- Reference pose data (sample images) must be prepared in advance and stored in the `base_data/` directory.

## 8. Requirements

- Python 3.8 or higher
- TensorFlow
- OpenCV
- scikit-learn
- movenet library

## 9. References

- [Movenet GitHub Repository](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection)
- [OpenCV Official Documentation](https://docs.opencv.org/)

## 10. Author

- Author: Magicherry
- Version: v0.5.1
- Last updated: 2024.4.1
