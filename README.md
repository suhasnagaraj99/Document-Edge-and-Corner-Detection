# Document-Edge-and-Corner-Detection

## Project Description
This repository contains code for ENPM667 Assignment 2 Question 1 - A video processing pipeline to find the 4 corners of the paper using Edge Detection, Hough Transform and Corner Detection.

![Video GIF](https://github.com/suhasnagaraj99/Document-Edge-and-Corner-Detection/blob/main/Q1_Results/suhas99_project2.gif)

### Required Libraries
Before running the code, ensure that the following Python libraries are installed:

- `cv2`
- `numpy`
- `matplotlib`

You can install if they are not already installed:

```bash
sudo apt-get install python3-opencv python3-numpy python3-matplotlib
```

### Running the Code
Follow these steps to run the code:

1. Make sure the video `project2_video1.mp4` is pasted in the same directory as the `suhas99_project2_project1.py` file.
2. Execute the `suhas99_project2_project1.py`; run the following command in your terminal:

```bash
python3 suhas99_project2_project1.py
```
3. The script uses the concepts of Canny Edge Detection, Hough Lines and Haris Corner Detector to compute and filter out paper edges and corners. It stores the video in form of a video `suhas99_project2.mp4`
