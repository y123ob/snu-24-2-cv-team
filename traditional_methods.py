# Stair Detection Using OpenCV

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import sqrt

# Variable to count the number of stairs
i = 0
# Variable to store the length of a line
b = 0
# Variable to store the count of unique line lengths
c = 0
# List to store the lengths of detected lines
a = []

# Read the image
image = cv2.imread("stair4.jpg")

# Convert the image to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection
edges = cv2.Canny(grayscale, 50, 150)

# Detect lines in the image using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 60, 3)

# Use try block to handle cases where no lines are detected
try:
    # Check if lines are detected
    if lines is not None:
        # Iterate through each detected line
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Ensure (x2 - x1) is not zero to avoid division by zero
                if x2 - x1 != 0:
                    # Calculate the slope of the line
                    slope = (y2 - y1) / (x2 - x1)
                    # Check if the line is almost horizontal
                    if 0 <= slope <= 0.5:
                        # Calculate the length of the line
                        b = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        a.append(b)
                        c = len(set(a))
                        i += 1
                        # Draw the line if it satisfies the conditions
                        if c != 1 and 4 < i < 20:
                            cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 2)
    # Check final conditions and output the result
    if c != 1 and 4 < i < 20:
        print("Stairs have been detected.")
    else:
        print("No stairs detected.")
except Exception:
    # Handle cases where no lines are detected
    print("No stairs detected.")

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off') 
plt.show()
