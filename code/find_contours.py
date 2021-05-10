# draw the bounding box around the forged area and find the center of it
# Mltimedia-lab
# Import required packages:
import cv2
import numpy as np
from  numpy import array

# Load the segmentation mask and convert it to grayscale:
image = cv2.imread("./test_mask.png")   
image[image == 255] = 100
image[image == 0] = 255
image[image == 100] = 0


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
# Apply cv2.threshold() to get a binary image
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# Find contours:
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw contours:
cv2.drawContours(image, contours, 0, (0, 255, 0), 2)

# Calculate image moments of the detected contour
M = cv2.moments(contours[0])

# Print center (debugging):
print("center X : '{}'".format(round(M['m10'] / M['m00'])))
print("center Y : '{}'".format(round(M['m01'] / M['m00'])))

# Draw a circle based centered at centroid coordinates
cv2.circle(image, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 5, (0, 255, 0), -1)

cv2.imwrite('contour_test.png',image)
a = array(contours)

# Show the segmentation mask with bounding box and center of it:
cv2.imshow("outline contour & centroid", image)

# blure the entire of RGB image
# load RGB image
#image_RGB = cv2.imread("./test.png")   #for rgb image
# blur the entire of image
#blur = cv2.GaussianBlur(image_RGB,(3,3),0)
# save the blurred image
#cv2.imwrite('total_blurred_test.png',blur)
