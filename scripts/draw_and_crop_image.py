import sys
import cv2
import numpy as np

drawing = False
ix, iy = -1, -1

def draw(event, x, y, flags, param):
    global ix, iy, drawing, img, DEFAULT

    if event == cv2.EVENT_LBUTTONDOWN:

        ix, iy = x, y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:

            img = DEFAULT.copy()
            cv2.rectangle(img,(ix,iy),(x,y),(0,180,255),1)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False
        crop(ix, iy, x, y)

def crop(ix, iy, x, y):
    global DEFAULT, imgCropped

    if (x < ix):
        tempx = x
        x = ix
        ix = tempx

    if (y < iy):
        tempy = y
        y = iy
        iy = tempy

    if ((ix != x) & (iy != y)):
        imgCropped = DEFAULT[iy:y, ix:x]
        cv2.imshow('imgCropped', imgCropped)

if (len(sys.argv) > 1):
	filename = sys.argv[1]
else:
	print('Pass a filename as first argument')
	sys.exit(0)

img = cv2.imread(filename)
DEFAULT = img.copy()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw)

while(1):
    cv2.imshow('Image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('r'):
        img = DEFAULT
        cv2.destroyWindow('imgCropped')
        print("Restored image")
    elif k == ord('g'):
        cv2.imwrite("imgCropped.png", imgCropped)
        print ("Saved image")

cv2.destroyAllWindows()
