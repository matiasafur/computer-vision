import cv2
import numpy as np
import sys

if (len(sys.argv) > 1):
    filename = sys.argv[1]
else:
    print('Pass a filename as first argument')
    sys.exit(0)

img1 = cv2.imread(filename, cv2.IMREAD_COLOR)
img2 = img1.copy()
DEFAULT = img1.copy()
DEFAULT2 = img2.copy()

def get_points_img1(event, x, y, flags, param):
    global img1_points, img1_click_n, img1, DEFAULT

    if event == cv2.EVENT_LBUTTONDOWN:

        if img1_click_n == 3:
            img1 = DEFAULT.copy()
            img1_points = [(-1,-1), (-1,-1), (-1,-1)]
            img1_click_n = 0

        if img1_click_n == 0:
            img1_points[0] = (x, y)

        elif img1_click_n == 1:
            img1_points[1] = (x, y)

        elif img1_click_n == 2:
            img1_points[2] = (x, y)

        cv2.circle(img1, (x, y), 5, (255,0,200), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1, str(img1_click_n+1), (x+5, y+4), font, 0.5, (0,0,0), 1, cv2.LINE_AA)

        img1_click_n += 1

    if event == cv2.EVENT_RBUTTONDOWN:

        img1 = DEFAULT.copy()
        img1_points = [(-1,-1), (-1,-1), (-1,-1)]
        img1_click_n = 0

def get_points_img2(event, x, y, flags, param):
    global img2_points, img2_click_n, img2, DEFAULT2

    if event == cv2.EVENT_LBUTTONDOWN:

        if img2_click_n == 3:
            img2 = DEFAULT.copy()
            img2_points = [(-1,-1), (-1,-1), (-1,-1)]
            img2_click_n = 0

        if img2_click_n == 0:
            img2_points[0] = (x, y)

        elif img2_click_n == 1:
            img2_points[1] = (x, y)

        elif img2_click_n == 2:
            img2_points[2] = (x, y)

        cv2.circle(img2, (x, y), 5, (255,0,200), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2, str(img2_click_n+1), (x+5, y+4), font, 0.5, (0,0,0), 1, cv2.LINE_AA)

        img2_click_n += 1

    if event == cv2.EVENT_RBUTTONDOWN:

        img2 = DEFAULT2.copy()
        img2_points = [(-1,-1), (-1,-1), (-1,-1)]
        img2_click_n = 0

img1_click_n = 0
img1_points = [(-1,-1), (-1,-1), (-1,-1)]
img2_click_n = 0
img2_points = [(-1,-1), (-1,-1), (-1,-1)]

cv2.namedWindow('image1')
cv2.namedWindow('image2')
cv2.setMouseCallback('image1', get_points_img1)
cv2.setMouseCallback('image2', get_points_img2)

while(1):

    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

    if (img1_click_n == 3 & img2_click_n == 3):

        pts1 = np.float32([img1_points[0], img1_points[1], img1_points[2]])
        pts2 = np.float32([img2_points[0], img2_points[1], img2_points[2]])
        M = cv2.getAffineTransform(pts1, pts2)
        img1_h, img1_w, img1_c = np.shape(img1)
        cv2.warpAffine(DEFAULT2, M, (img1_w, img1_h), img1)

cv2.destroyAllWindows()
