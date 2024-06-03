import sys
import cv2

if (len(sys.argv) > 1):
	filename = sys.argv[1]
else:
	print('Pass a filename as first argument')
	sys.exit(0)

cap = cv2.VideoCapture(filename)

fps = cap.get(cv2.CAP_PROP_FPS)

time = int(1000/(fps))

print ('FPS from video: ', fps)

while (cap.isOpened()):


	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

	cv2.imshow('frame', gray)

	if ((cv2.waitKey(time) & 0xFF) == ord('q')):
	
		break

cap.release()
cv2.destroyAllWindows()
