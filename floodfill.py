
#import numpy
#im = Image.open("sample2.png")
#np_im = numpy.array(r)
#np_im.resize((20,2,3))

#print (np_im.shape)


# floodfill for mask?
import cv2
import numpy as np


img = cv2.imread("./document.jpg")
img_copy = img.copy()

#Improvised way to find the Off White color (it's working because the Off White has the maximum color components values).
tmp = cv2.dilate(img, np.ones((50,50), np.uint8), iterations=10)

# Color of Off-White pixel
offwhite = tmp[0, 0, :]

# Convert to tuple
offwhite = tuple((int(offwhite[0]), int(offwhite[1]), int(offwhite[2])))

# Fill black pixels with off-white color
cv2.floodFill(img_copy, None, seedPoint=(0,0), newVal=offwhite)
