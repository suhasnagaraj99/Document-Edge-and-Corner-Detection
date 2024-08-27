# -*- coding: utf-8 -*-

"""
# Problem 1

Design a video processing pipeline to find the 4 corners of the paper using Edge Detection, Hough Transform and
Corner Detection. Overlay the boundary(edges) of the paper and highlight the four corners of the paper. (Note: that you must remove any frames
which are blurry by using Variance of Laplacian and report the number of blurry frames removed).

"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math

## 1) Read the video and extract individual frames using OpenCV.

# creating a video file object
video=cv.VideoCapture("project2_video1.mp4")

# Condition statement to check if the video is opened and loaded, without errors
if video.isOpened():
  print("Video is Opened",'\n')

  # Extracting video properties to match with the output video
  # width of each frame in the video
  width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
  # height of each frame in the video
  height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

else:
  print("Error")

"""The below cell is the main cell which performs all the operations on the frames extracted from the video.

Pipeline:
1. Extract individual frames from the video
2. Convert the frame to grayscale
3. Filter out the blurry frames by taking varience of Laplacian
4. The remaining frames are blurred and segmented
5. Edges are detected from the segmented frame
6. From the binary image of edges, major straight lines are extracted
7. The straight lines are used to obtain the intersections
8. The straight lines represent the edges of the paper whereas the intersections represent the corners.
9. The obtained corners are then verified by using Harris Corner Detector. Only those points which are obtained as both intersections and from Harris detector are selected for plotting.
10. The edges and corners are plotted on the frame.
11. The frames are written on to an output video
"""

# creating a output video object
# video codec
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('suhas99_project2.mp4', fourcc, 10, (width, height))

# indexing variables to count number of blurry frames and total number of frames
n1,n2 = 0,0
while video.isOpened():
  ret, frame = video.read()
  n1=n1+1
  if ret == False:
    break

  # converting the frames to greyscale
  gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


  ## 2) Skip blurry frames (use Variance of the Laplacian and decide a suitable threshold)
  ## Note: Any value below 150 for the Variance of the Laplacian, suggests that it’s a blurry image.

  ## Using the Laplacian kernel, cv.filter2D and np.var to find the varience of the Laplacian
  laplacian_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
  laplacian = cv.filter2D(gray_frame, cv.CV_64F, laplacian_kernel)
  variance = np.var(laplacian)

  ## conditional statement to skip blurry frames
  if variance < 110:
      n2 = n2 + 1
      continue

  ## 3) Segment out the unwanted background area (example: convert to gray scale and keep white regions)
  # The grayscale frame is first blurred so only major/strong features are considered for further processing
  blurred = cv.GaussianBlur(gray_frame, (3, 3), 0)
  # The blurred frame is then segmented to extract pixels which have grayscale value of above 200
  segmented_frame = np.where(blurred > 200, 255, 0).astype(np.uint8)

  ## 4) Detect edges pixels in each frame (you can use any edge detector)
  ## Canny edge detector is used to detect edges
  edge_frame = cv.Canny(segmented_frame, 50, 150)

  ## 5) Use the detected edge pixels to extract straight lines in each frame (hint: use Hough Transform)
  ## 6) Filter out “short” lines and only keep a few dominant lines.
  ## cv.HoughLinesP is used to detect lines and maxLineGap and minLineLength parameters are used to filter out short lines
  lines = cv.HoughLinesP(edge_frame , rho = 1 , theta = np.pi/180 , threshold = 65 , maxLineGap = 9 , minLineLength = 110)

  ## 7) Compute the intersections of the Hough Lines – these are the putative corners of the paper.
  ## Each line detected is represented as a equation and intersections are computed.
  ## As paper is a rectangle, a conditional statement is used to filter out corners based on the angle between the edges
  intersection_points = []
  for i in range(len(lines)):
      for j in range(i+1, len(lines)):
          line1 = lines[i]
          line2 = lines[j]
          x1, y1, x2, y2 = line1[0]
          x3, y3, x4, y4 = line2[0]

          # Slopes
          m1 = (y2 - y1) / (x2 - x1)
          m2 = (y4 - y3) / (x4 - x3)

          # y-intercepts
          c1= y1 -(m1*x1)
          c2= y3 -(m2*x3)

          # angle between a pair of lines
          angle = math.atan(abs((m2 - m1) / (1 + m1 * m2)))
          angle_degrees = math.degrees(angle)

          ## As paper is a rectangle, a conditional statement is used to filter out corners based on the angle between the edges/lines
          if angle_degrees < 120 and angle_degrees > 60:
            x=int((c2-c1)/(m1-m2))
            y=int((m1*x)+c1)
            intersection_points.append((x, y))

  ## 8) Verify the existence of those corners with Harris corner detector. (use OpenCV built-in function)
  dst = cv.cornerHarris(gray_frame, blockSize=2, ksize=3, k=0.04)
  threshold = 0.25 * dst.max()
  corner_coordinates = np.argwhere(dst > threshold)
  ## The detected corners are verified by overlaying/overlapping them with the corners detected by Harris Corner Detector and considering. Only the overlapped corners are considered.
  overlapping_points = []
  for intersection_point in intersection_points:
      for corner in corner_coordinates:
          # distance between the computed corner and detected corner. If the distance is less than 3, then the computed corner is considered as overlapping with Harris Corner
          distance = np.linalg.norm(np.array(intersection_point) - corner[::-1])
          if distance < 15:
              overlapping_points.append(intersection_point)
              break
  ## array which stores the corners verified by harris corner detector
  overlapping_points = np.array(overlapping_points)

  ## Creating a copy of overlapping points to further filter them based on distance/proximity
  filtered_points = list(overlapping_points)
  num_points = len(filtered_points)
  i = 0
  ## Loop for filtering out points based on relative distance
  while i < num_points:
    j = i + 1
    while j < num_points:
      ## Filtering out points which are close to each other
      dist = math.sqrt((filtered_points[i][0] - filtered_points[j][0])**2 + (filtered_points[i][1] - filtered_points[j][1])**2)
      if dist < 15:
        ## if a point is very close to another point, one point is deleted. This makes sure that a corner is not represented by 2 points
        del filtered_points[j]
        num_points -= 1
      else:
        j += 1
    i += 1


  ## Drawing the lines/edges detected
  for j in lines:
    for i in j:
      ## Filtering out small lines
      dist = math.sqrt((i[0] - i[2])**2 + (i[1] - i[3])**2)
      if dist<130:
        ## Small lines are not drawn
        continue
      cv.line(frame, (i[0], i[1]), (i[2] , i[3]), (255, 0, 0), 3)

  ## Drawing the corners detected
  for vertex in filtered_points:
    cv.circle(frame, vertex, 5, (0, 0, 255), -1)

  ## Writing the frame to out
  out.write(frame)

out.release()

## Printing the total frames, number of frames skipped and number of frames considered
print("Total number of frames: ",n1,'\n')
print("Number of frames skipped (Blurry): ",n2,'\n')
print("Number of frames considered: ",n1-n2,'\n')

"""The number of frames considered is more than 50% of the total frames"""
