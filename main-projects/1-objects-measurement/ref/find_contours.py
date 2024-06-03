import numpy as np
import cv2

chessboardSize = (4, 3)
threshold = 100

# I assume that the top-left corner is the first corner of the pattern
paperSize_h = 253.3 # mm 
paperSize_v = 190 # mm

mmpx = 0.3958

# 253.3mm-------------- 640px
# 52mm   -------------- 131.38px

# 190mm ---------------- 480px
# 35mm  ---------------- 88.42px

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def getContours(frame):
    global threshold

    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray scale
    imblur = cv2.blur(imgray, (10,10)) # Blur filter to reduce noise
    ret, imthresh = cv2.threshold(imblur, threshold, 255, cv2.THRESH_BINARY) # Threshold to convert the image to B&W
    
    # Patch
    cv2.rectangle(imthresh, (400, 0), (640, 150), (255,255,255), -1)
    
    # Calculate contours
    contours, hierarchy = cv2.findContours(imthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    return contours, frame

def rectify(frame):
    global n, h, v
    
    drawedFrame = frame.copy()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Gray scale
    ret, corners = cv2.findChessboardCorners(grayFrame, chessboardSize, None, cv2.CALIB_CB_FAST_CHECK)
    
    if ret is True:
        refinedCorners = cv2.cornerSubPix(grayFrame, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, chessboardSize, refinedCorners, ret)
        
        imgSize = (frame.shape[1], frame.shape[0])
        
        # Points to map
        src = np.float32([corners[0], corners[3], corners[8], corners[11]])
        dst = np.float32([[508, 0], [640, 0], [508, 88], [640, 88]]) 
        
        cv2.circle(frame, tuple(src[0][0]), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(src[1][0]), 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(src[2][0]), 5, (255, 0, 0), -1)
        cv2.circle(frame, tuple(src[3][0]), 5, (0, 255, 255), -1)
        
        cv2.circle(frame, tuple(dst[0]), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(dst[1]), 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(dst[2]), 5, (255, 0, 0), -1)
        cv2.circle(frame, tuple(dst[3]), 5, (0, 255, 255), -1)
        
        M = cv2.getPerspectiveTransform(src, dst) 
        rectifiedFrame = cv2.warpPerspective(drawedFrame, M, imgSize)

        return True, rectifiedFrame
    else: 
        return False, frame

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # Flip horizontally

    valid, rectifiedFrame = rectify(frame)
    
    if valid:
        contours, drawedFrame = getContours(rectifiedFrame.copy())

        # Measurement
        for contour in contours:
            x, y, contourW, contourH = cv2.boundingRect(contour) # Rectangle within which the object to be measured is located
            cv2.line(drawedFrame, (x, y + contourH), (x + contourW, y + contourH), (0,0,255), 2) # Width
            cv2.line(drawedFrame, (x + contourW, y), (x + contourW, y + contourH), (0,0,255), 2) # Height
            # Convert to mm
            contourW_mm = round(mmpx * contourW, 2)
            contourH_mm = round(mmpx * contourH, 2)
            # Labels
            cv2.putText(drawedFrame, str(contourW_mm) + "mm", (round(x + contourW / 2), y + contourH + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.putText(drawedFrame, str(contourH_mm) + "mm", (x + contourW, round(y + contourH / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    else:
        drawedFrame = frame.copy()

    cv2.namedWindow('Output')
    cv2.imshow('Output', drawedFrame)
    cv2.namedWindow('Capture')
    cv2.imshow('Capture', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('+'):
        threshold = threshold + 1
        print(threshold)
    if key == ord('-'):
        threshold = threshold - 1
        print(threshold)
    
cap.release()
cv2.destroyAllWindows()
