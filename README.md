## Document Scanner using Edge Detection and Perspective Warping

A computer vision project that detects a document or paper within an image and transforms it into a clean, top-down scanned view. The program uses Canny edge detection to locate the document's boundary, identifies its four corners through contour analysis, and applies a perspective warp to flatten the page, finishing with an adaptive threshold for a crisp black-and-white scanned look.

## Features

- Document boundary detection using Canny edge detection
- Automatic four-corner contour identification
- Corner ordering for accurate perspective transformation
- Perspective warp to produce a flat, top-down document view
- Black-and-white scanned output using adaptive thresholding
- File picker dialog for easy image selection

## Technologies Used

- Python 
- OpenCV 
- NumPy
- Tkinter 

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Hp2806/document-scanner.git
```
