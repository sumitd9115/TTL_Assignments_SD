# TTL Assignment 07 — Pedestrian Detection using OpenCV (HOG + SVM)

## Overview
This notebook demonstrates pedestrian detection using OpenCV's built-in Histogram of Oriented Gradients (HOG) descriptor combined with a pre-trained linear SVM people detector.

## What the notebook does
1. **HOG detector setup** — Initializes `cv2.HOGDescriptor()` and loads OpenCV's built-in pre-trained people detector via `cv2.HOGDescriptor_getDefaultPeopleDetector()`.
2. **Synthetic street scene generation** — Builds a simple synthetic "street scene" image containing person-shaped silhouettes (rectangle body + circular head) at random positions, standing in for a real street/CCTV frame, and displays it.
3. **Detection** — Runs `hog.detectMultiScale()` on the frame to detect pedestrian-like regions, printing each detected bounding box and its confidence score.
4. **Visualization** — Draws detected boxes (green) alongside the ground-truth silhouette boxes (red) on the image for visual comparison.
5. **Real-world usage note** — Explains that swapping the synthetic frame for `cv2.imread()` on a real image or a `cv2.VideoCapture()` frame lets the exact same pipeline run on real pedestrian footage.

## Key libraries
`opencv-python` (`cv2`), `numpy`, `matplotlib`

## Outcome
A working HOG+SVM detection pipeline that draws bounding boxes around detected pedestrian-like shapes, ready to be pointed at real images or video frames.

## Important note
The "street scene" here is **synthetically generated** (simple silhouettes), since no real street image was available in this environment. OpenCV's default HOG+SVM detector is trained on real human shapes, so detection performance on these synthetic silhouettes may be limited — for a real demo/interview, run this same code on an actual photo or webcam frame containing people for a meaningful result.

## How to run
Run all cells top to bottom in a Jupyter environment with `opencv-python`, `numpy`, and `matplotlib` installed. No external dataset or internet access is required.
