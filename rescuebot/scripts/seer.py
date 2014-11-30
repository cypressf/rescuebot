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

    lower_red = np.array([150, 55, 55])
    upper_red = np.array([200, 255, 255])
    mask_red = cv2.inRange(hsv_img, lower_red, upper_red)

    lower_blue = np.array([100, 55, 55])
    upper_blue = np.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)

    lower_yellow = np.array([20, 55,55])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    lower_green = np.array([60, 55, 55])
    upper_green = np.array([100, 255, 255])
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    location = None
    global ball_location
    if circles is not None:
    	for c in circles[0,:]:
			ROI_red = mask_red[c[1]-c[2]:c[1]+c[2], c[0]-c[2]:c[0]+c[2]]
			mean_red = cv2.mean(ROI_red)

			ROI_green = mask_green[c[1]-c[2]:c[1]+c[2], c[0]-c[2]:c[0]+c[2]]
			mean_green = cv2.mean(ROI_green)

			ROI_yellow = mask_yellow[c[1]-c[2]:c[1]+c[2], c[0]-c[2]:c[0]+c[2]]
			mean_yellow = cv2.mean(ROI_yellow)

			ROI_blue = mask_blue[c[1]-c[2]:c[1]+c[2], c[0]-c[2]:c[0]+c[2]]
			mean_blue = cv2.mean(ROI_blue)

			cv2.imshow('mask', mask_blue)

			cv2.circle(img_out,(c[0],c[1]),c[2],(0,255,0),1)
			# draw the center of the circle
			cv2.circle(img_out,(c[0],c[1]),2,(0,0,255),3)

			if mean_yellow[0] > 200:
				#print mean
				cv2.rectangle(img_out, (c[0]-c[2], c[1]+c[2]), (c[0]+c[2],c[1]-c[2]) , (0,255,255), 2)
				# draw the outer circle
				cv2.circle(img_out,(c[0],c[1]),c[2],(0,255,255),5)
				# draw the center of the circle
				cv2.circle(img_out,(c[0],c[1]),2,(0,255,255),3)
				location = Vector3(c[0], c[1],c[2])
				#print (c[0],c[1],c[2])
				ball_location.append(location)

			if mean_blue[0] > 50:
				#print mean
				cv2.rectangle(img_out, (c[0]-c[2], c[1]+c[2]), (c[0]+c[2],c[1]-c[2]) , (255,0,0), 2)
				# draw the outer circle
				cv2.circle(img_out,(c[0],c[1]),c[2],(255,0,0),5)
				# draw the center of the circle
				cv2.circle(img_out,(c[0],c[1]),2,(255,0,0),3)
				location = Vector3(c[0], c[1],c[2])
				#print (c[0],c[1],c[2])
				ball_location.append(location)

			if mean_red[0] > 100:
				#print mean
				cv2.rectangle(img_out, (c[0]-c[2], c[1]+c[2]), (c[0]+c[2],c[1]-c[2]) , (0,0,255), 2)
				# draw the outer circle
				cv2.circle(img_out,(c[0],c[1]),c[2],(0,0,255),5)
				# draw the center of the circle
				cv2.circle(img_out,(c[0],c[1]),2,(0,0,255),3)
				location = Vector3(c[0], c[1],c[2])
				#print (c[0],c[1],c[2])
				ball_location.append(location)

			if mean_green[0] > 50:
				#print mean
				cv2.rectangle(img_out, (c[0]-c[2], c[1]+c[2]), (c[0]+c[2],c[1]-c[2]) , (0,255,0), 2)
				# draw the outer circle
				cv2.circle(img_out,(c[0],c[1]),c[2],(0,255,0),5)
				# draw the center of the circle
				cv2.circle(img_out,(c[0],c[1]),2,(0,255,0),3)
				location = Vector3(c[0], c[1],c[2])
				#print (c[0],c[1],c[2])
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

