#Playing from a file:
import numpy as np
import cv2

cap = cv2.VideoCapture('file:///home/victoria/rescuebot_bagfiles/images_3/output.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    if cv2.waitKey(120) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()