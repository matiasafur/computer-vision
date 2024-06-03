import numpy as np
import cv2
import sys

def translate (image, x, y):
	
	(h, w) = image.shape[:2]
	
	M = np.float32 ([[1, 0, x], [0, 1, y]])

	shifted = cv2.warpAffine (image, M, (h, w))

	return shifted

def rotate (image, center=None, angle = 0, scale = 1.0):
	
	if center is None:

		(h, w) = image.shape [:2]	
		center = (w/2, h/2)

	M = cv2.getRotationMatrix2D (center, angle, scale)

	rotated = cv2.warpAffine (image, M, (w, h))
	
	return rotated


if (len(sys.argv) > 1):
	filename = sys.argv[1]
else:
	print('Pass a filename as first argument')
	sys.exit(0)

px = 50
py = 50
center = None
angle = 5
scale = 0.5

img = cv2.imread(filename)
img_shifted = translate(img, px, py)
img_rotated = rotate(img_shifted, center, angle, scale)

while(1):

    cv2.imshow('Image', img_rotated)

    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break

cv2.destroyAllWindows()

