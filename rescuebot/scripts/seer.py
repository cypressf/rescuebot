#Playing from a file:
import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3

ball_location = []
cap = cv2.VideoCapture('file:///home/victoria/rescuebot_bagfiles/images_3/output.avi')

def find_circles(img_src,img_out):
    """Finds and plots circles using Hough Circle detection."""
    circles = cv2.HoughCircles(img_src, cv2.cv.CV_HOUGH_GRADIENT, 1, img_src.shape[0]/8, param1=10, param2=30, minRadius=50, maxRadius=75)
    hsv_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 55, 55])
    upper_red = np.array([0, 255, 255])
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    location = None
    if circles is not None:
    	for c in circles[0,:]:
	    	ROI = mask[c[1]-c[2]:c[1]+c[2], c[0]-c[2]:c[0]+c[2]]
	      	mean = cv2.mean(ROI)
	      	# print mean
	      	cv2.imshow('mask', mask)
	      	# cv2.imshow('canny', img_src)
	      # 	if cv2.waitKey(10) & 0xFF == ord('q'):
		    	# cap.release()
		    	# cv2.destroyAllWindows()   
	      	cv2.circle(img_out,(c[0],c[1]),c[2],(0,255,0),1)
	      	# draw the center of the circle
	      	cv2.circle(img_out,(c[0],c[1]),2,(0,0,255),3)
	      	if mean[0] > 10:
	      		print mean
	      	#if mask[c[1], c[0]]  > 100:
	      		cv2.rectangle(img_out, (c[0]-c[2], c[1]+c[2]), (c[0]+c[2],c[1]-c[2]) , (255,0,0), 2)
		    	# draw the outer circle
		    	cv2.circle(img_out,(c[0],c[1]),c[2],(255,0,0),5)
		        # draw the center of the circle
		    	cv2.circle(img_out,(c[0],c[1]),2,(0,0,255),3)
		    	location = Vector3(c[0], c[1],c[2])
		        #print (c[0],c[1],c[2])
    global ball_location
    ball_location.append(location)

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 5, 50)
    find_circles(edges,frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(120) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

