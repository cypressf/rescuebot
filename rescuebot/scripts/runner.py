#!/usr/bin/env python
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image, LaserScan
import rospy
import cv_bridge
import sys
import signal

class image_converter:
  def __init__(self):
    self.image_pub = rospy.Publisher("/processed_image",Image)
    self.target_pub = rospy.Publisher("/target_coords",Vector3)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_raw",Image,self.callback)
    self.target_location = None

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e

    #TODO Image Processing

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      if self.target_location:
        self.target_pub.publish(self.target_location)
    except CvBridgeError, e:
      print e


class Controller:
    def __init__(self):
        rospy.init_node('controller', anonymous=True)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.img_sub = None
        self.laser_scan_sub = None
        self.bridge = cv_bridge.CvBridge()
        self.running = False
        self.cmd_vel = Twist()
        signal.signal(signal.SIGINT, self.stop_neato)

        self.valid_ranges = []

    def image_received(self, image_message):
        """Process image from camera(s)"""
        cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
        # TODO

    def laser_scan_received(self, laser_scan_message):
        """Process laser scan points from LiDAR"""
        #Currently just appends valid numbers
        self.valid_ranges = []
        for i in range(5): #if it sees anything within 5 meters, it is valid, throwout greater values
            if msg.ranges[i] > 0 and msg.ranges[i] < 360: #You can make this any range..
                valid_ranges.append(msg.ranges[i])

    def run(self):
        """Subscribe to the laser scan data and images."""
        self.running = True
        self.laser_scan_sub = rospy.Subscriber('scan', LaserScan, self.laser_scan_received)
        self.img_sub = rospy.Subscriber('camera/image_raw', Image, self.image_received)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.running:
            self.pub.publish(self.cmd_vel)
            rate.sleep()

    def stop_neato(self, signal, frame):
        """Stop the neato in case of signal interrupt."""
        self.running = False
        self.pub.publish(Twist())
        sys.exit(0)


if __name__ == "__main__":
    controller = Controller()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
