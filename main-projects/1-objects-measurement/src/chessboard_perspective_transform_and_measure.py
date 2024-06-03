import cv2
import numpy as np
import sys

# Conversion from px to mm
# 269mm/640px = 0,4203
relation = 0.4203
threshold = 100

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    frame_h, frame_w, frame_c = np.shape(frame)
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(frame_gray, (3, 4), None, cv2.CALIB_CB_FAST_CHECK)

    if ret is True:
        corners2 = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, (3, 4), corners2, ret)

        margin = 50
        dst_w = 128
        dst_h = 85
        dst_square = [(frame_w - margin, margin), (frame_w - margin, margin + dst_h), (frame_w - margin - dst_w, margin), (frame_w - margin - dst_w, margin + dst_h)]
        # dst_square = [(639, 0), (639, 83), (439, 0), (439, 83)]

        src = np.float32([corners[0], corners[2], corners[9], corners[11]])
        dst = np.float32([dst_square[0], dst_square[1], dst_square[2], dst_square[3]])

        M = cv2.getPerspectiveTransform(src, dst)
        cv2.warpPerspective(frame, M, (frame_w, frame_h), frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gauss = cv2.GaussianBlur(frame_gray, (5, 5), 2)
        ret, frame_thresh = cv2.threshold(frame_gauss, threshold, 255, cv2.THRESH_BINARY)

        # Find and draw contours
        cv2.rectangle(frame_thresh, (400, 0), (640, 220), (255, 255, 255), -1)
        contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

        for contour in contours:
            x, y, contour_w, contour_h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + contour_w, y + contour_h), (150, 0, 150), 2)
            measure_w = contour_w * relation
            cv2.putText(frame, str(measure_w), (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,0,150), 1, cv2.LINE_AA)

    cv2.imshow('image_1', frame)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break
    if k == ord('+'):
        threshold = threshold + 1
        print (threshold)
    if k == ord('-'):
        threshold = threshold - 1
        print (threshold)

cap.release()
cv2.destroyAllWindows()
