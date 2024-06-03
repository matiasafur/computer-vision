import sys
import cv2
import time

if (len(sys.argv) > 1):
    filename = sys.argv[1]
else:
    print('Pass a filename as first argument')
    sys.exit(0)

cap = cv2.VideoCapture(filename)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

# obtencion de fps
fps = cap.get(cv2.CAP_PROP_FPS)

# obtencion de ancho y alto del video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# se escrie un archivo de video de salida
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
t = int(1000/fps)

while (cap.isOpened()):

    ret, frame = cap.read()

    if ret is True:

        start = time.time()

        out.write(frame)

        cv2.imshow('frame', frame)

        stop = time.time()

        delta = int((stop - start) * 1000)

        # la duracion del video es hasta que el programa terminao hasta que se precione la tecla 'q'
        if (delta < t):

            if ((cv2.waitKey(t - delta)  & 0xFF) == ord('q')):
                break

        else:
            break

cap.release()
cv2.destroyAllWindows()
