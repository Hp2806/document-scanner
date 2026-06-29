import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# Hide the main tkinter window
Tk().withdraw()

# Open file picker dialog
file_path = askopenfilename(
    title="Select Document Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

# If user cancels 
if not file_path:
    print("No image selected!")
    exit()

# Load selected image
image = cv2.imread(file_path)
if image is None:
    print("Error: Could not load the selected image!")
    exit()
orig = image.copy()

# Resize for consistent contour detection (keeps a copy of the ratio to scale back up later)
height = 700
ratio = height / image.shape[0]
width = int(image.shape[1] * ratio)
resized = cv2.resize(image, (width, height))

# Convert to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection to find the document boundary
edged = cv2.Canny(blur, 75, 200)

# Dilate to close small gaps in the edges
kernel = np.ones((3, 3), np.uint8)
edged = cv2.dilate(edged, kernel, iterations=1)

# --- Find document contour ---
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
doc_contour = None
for cnt in contours:
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    if len(approx) == 4:  # found a 4-sided shape, assume it's the document
        doc_contour = approx
        break
if doc_contour is None:
    print("No 4-sided document contour found! Try a clearer image with good contrast.")
    exit()

contour_vis = resized.copy()
cv2.drawContours(contour_vis, [doc_contour], -1, (0, 255, 0), 2)

# --- Order the 4 points ---
pts = doc_contour.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]   # top-left 
rect[2] = pts[np.argmax(s)]   # bottom-right 
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]  # top-right 
rect[3] = pts[np.argmax(diff)]  # bottom-left 

# Scale points back up to the original full-resolution image
rect = rect / ratio
(tl, tr, br, bl) = rect

# --- Compute width and height of the warped output ---
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# Destination points for the flattened, top-down document
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")

# --- Perspective warp on the original full-resolution image ---
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
# Convert warped result to a clean black & white "scanned" look
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped_bw = cv2.adaptiveThreshold(warped_gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 10)
cv2.imshow("Edged Image", edged)
cv2.imshow("Detected Contour", contour_vis)
cv2.imshow("Warped Color Scan", warped)
cv2.imshow("Final B&W Scan", warped_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

