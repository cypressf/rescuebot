#!/usr/bin/env python
import roslib
roslib.load_manifest('neato_fetch')
import rospkg
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Vector3
import math
import numpy as np

class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("/processed_image",Image)
    self.ball_pub = rospy.Publisher("/ball_coords",Vector3)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_raw",Image,self.callback)
    self.ball_location = None

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    #Image Processing
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 5, 50)
    self.find_circles(edges,cv_image)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      if self.ball_location:
        self.ball_pub.publish(self.ball_location)
    except CvBridgeError, e:
      print e

  def find_circles(self,img_src,img_out):
    """Finds and plots circles using Hough Circle detection."""
    circles = cv2.HoughCircles(img_src, cv2.cv.CV_HOUGH_GRADIENT, 1, img_src.shape[0]/8, param1=10, param2=30, minRadius=10, maxRadius=50)
    hsv_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 55, 55])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    location = None
    if circles is not None:
      for c in circles[0,:]:

          ROI = mask[c[0]-c[2]:c[0]+c[2], c[1]-c[2]:c[1]+c[2]]
          
          mean = cv2.mean(ROI)
          #print mean

          #cv2.imshow('mask', ROI)
          # cv2.imshow('canny', img_src)
          # if cv2.waitKey(50) & 0xFF == ord('q'):
          #   cap.release()
          #   cv2.destroyAllWindows()   

          # cv2.circle(img_out,(c[0],c[1]),c[2],(0,255,0),1)
          # # draw the center of the circle
          # cv2.circle(img_out,(c[0],c[1]),2,(0,0,255),3)       

          if mask[c[1], c[0]]  > 100:
            # draw the outer circle
            cv2.circle(img_out,(c[0],c[1]),c[2],(255,0,0),5)
            # draw the center of the circle
            cv2.circle(img_out,(c[0],c[1]),2,(0,0,255),3)
            location = Vector3(c[0], c[1],c[2])
            #print (c[0],c[1],c[2])
      self.ball_location = location


class ball_follower:
  def __init__(self):
    self.move_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    self.move_sub  = rospy.Subscriber('ball_coords', Vector3, self.coordinate_to_action)
    self.lin_vel = 0
    self.ang_vel = 0
    self.frame_height = 480
    self.frame_width = 640
  def coordinate_to_action(self, msg):
    x = msg.x
    y = msg.y
    r = msg.z

    depth_proportion = -0.025
    depth_intercept = 1.35
    depth = r*depth_proportion + depth_intercept
    # print depth
    y_transform = self.frame_height/2 - y
    x_transform = x-self.frame_width/2
    angle_diff = x_transform
    print angle_diff

    if (angle_diff-10) < 0 and (angle_diff + 10) > 0:
      angle_diff = 0 

    if depth-.05<0.5 and depth+.05>0.5:
      depth = 0.5

    # print "x: ", x_transform
    # print "y: ",y
    # print "d: ", depth
    # print "a: ", angle_diff

    twist = Twist()

    lin_proportion = -(0.5-depth)*0.07
    twist.linear = Vector3(lin_proportion, 0, 0)

    turn_proportion = -0.0005*(angle_diff)

    twist.angular = Vector3(0, 0, turn_proportion)

    self.move_pub.publish(twist.linear, twist.angular)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  fido = ball_follower()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"

if __name__ == '__main__':
    main(sys.argv)