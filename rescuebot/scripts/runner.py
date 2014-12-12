#!/usr/bin/env python
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Pose
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv_bridge
import sys
import signal
from tf import TransformListener
from tf.transformations import euler_from_quaternion
from math import cos, sin, pi, sqrt, ceil
from matplotlib.pyplot import imshow
import cv2
import numpy as np


class OccupancyGridMapper:
    """ Implements simple occupancy grid mapping """

    def __init__(self):
        cv2.namedWindow("map")

        #Establish mapping conventions
        self.origin = [-10, -10]
        self.seq = 0
        self.resolution = .1
        self.n = 200
        self.p_occ = .5
        self.odds_ratio_hit = 3.0
        self.odds_ratio_miss = .2
        self.odds_ratios = self.p_occ / (1 - self.p_occ) * np.ones((self.n, self.n))
        self.tf_listener = TransformListener()

        #Subscribers and Publishers
        rospy.Subscriber("scan", LaserScan, self.process_scan, queue_size=1)
        self.pub = rospy.Publisher("map", OccupancyGrid)
        self.coords_sub_red = rospy.Subscriber('ball_coords_red', Vector3, self.coordinate_to_map_red)
        self.coords_sub_green = rospy.Subscriber('ball_coords_green', Vector3, self.coordinate_to_map_green)
        self.coords_sub_blue = rospy.Subscriber('ball_coords_blue', Vector3, self.coordinate_to_map_blue)
        self.coords_sub_yellow = rospy.Subscriber('ball_coords_yellow', Vector3, self.coordinate_to_map_yellow)

        #Camera translations
        #TODO Add stuff for each color, so can map more than one at a time
        self.frame_height = 480
        self.frame_width = 640
        self.depth_proportion = -0.0025
        self.depth_intercept = 1.35

        self.red = (0, 0, 255)
        self.yellow = (0, 255, 255)
        self.blue = (255, 0, 0)
        self.green = (0, 255, 0)

        self.depth_yellow = 0
        self.y_transform_yellow = 0 
        self.x_transform_yellow = 0
        self.angle_diff_yellow = 0

        self.depth_red = 0
        self.y_transform_red = 0 
        self.x_transform_red = 0
        self.angle_diff_red = 0

        self.depth_green = 0
        self.y_transform_green = 0 
        self.x_transform_green = 0
        self.angle_diff_green = 0

        self.depth_blue = 0
        self.y_transform_blue = 0 
        self.x_transform_blue = 0
        self.angle_diff_blue = 0

        self.x_camera_red = None
        self.y_camera_red = None
        self.x_camera_blue = None
        self.y_camera_blue = None
        self.x_camera_green = None
        self.y_camera_green = None
        self.x_camera_yellow = None
        self.y_camera_yellow = None

    def is_in_map(self, x_ind, y_ind):
        """ Return whether or not the given point is within the map boundaries """
        return not (x_ind < self.origin[0] or
                    x_ind > self.origin[0] + self.n * self.resolution or
                    y_ind < self.origin[1] or
                    y_ind > self.origin[1] + self.n * self.resolution)

    # def process_color_and_coordinate(self, msg):
    #     """ Parse the Twist message into usable tupple, takes in data from color Subscriber"""
    #     tuple_color = (int(color.x), int(color.y), int(color.z))
    #     if tuple_color == (0, 0, 255):
    #         self.red = tuple_color
    #         self.coordinate_to_map_red(msg[color])
    #     elif tuple_color == (255, 0, 0):
    #         self.blue = tuple_color
    #         self.coordinate_to_map_blue(msg[color])
    #     elif tuple_color == (0, 255, 0):
    #         self.green = tuple_color
    #         self.coordinate_to_map_green(msg[color])
    #     elif tuple_color == (0, 255, 255):
    #         self.yellow = tuple_color
    #         self.coordinate_to_map_yellow(msg[color])

    def process_scan(self, msg):
        """ Callback function for the laser scan messages """
        if len(msg.ranges) <= 330:
            # throw out scans that don't have more than 90% of the data
            return
        # get pose according to the odometry
        p = PoseStamped(header=Header(stamp=msg.header.stamp, frame_id="base_link"), pose=Pose())
        self.odom_pose = self.tf_listener.transformPose("odom", p)
        # convert the odom pose to the tuple (x,y,theta)
        self.odom_pose = OccupancyGridMapper.convert_pose_to_xy_and_theta(self.odom_pose.pose)
        for i in range(len(msg.ranges)):
            if 0.0 < msg.ranges[i] < 5.0: #for any reding within 5 meters
                #Using the pose and the measurement nd the angle, find it in the world
                map_x = self.odom_pose[0] + msg.ranges[i] * cos(i * pi / 180.0 + self.odom_pose[2])
                map_y = self.odom_pose[1] + msg.ranges[i] * sin(i * pi / 180.0 + self.odom_pose[2])
                #Relate that map measure with a place in the picture
                x_detect = int((map_x - self.origin[0]) / self.resolution)
                y_detect = int((map_y - self.origin[1]) / self.resolution)
                #Determine how to mark the location in the map, along with the stuff inbetween
                u = (map_x - self.odom_pose[0], map_y - self.odom_pose[1])
                magnitude = sqrt(u[0] ** 2 + u[1] ** 2)
                n_steps = max([1, int(ceil(magnitude / self.resolution))])
                u_step = (u[0] / (n_steps - 1), u[1] / (n_steps - 1))
                marked = set()
                for i in range(n_steps):
                    curr_x = self.odom_pose[0] + i * u_step[0]
                    curr_y = self.odom_pose[1] + i * u_step[1]
                    if not (self.is_in_map(curr_x, curr_y)):
                        break

                    x_ind = int((curr_x - self.origin[0]) / self.resolution)
                    y_ind = int((curr_y - self.origin[1]) / self.resolution)
                    if x_ind == x_detect and y_ind == y_detect:
                        break
                    if not ((x_ind, y_ind) in marked):
                        # odds ratio is updated according to the inverse sensor model
                        self.odds_ratios[x_ind, y_ind] *= self.p_occ / (1 - self.p_occ) * self.odds_ratio_miss
                        marked.add((x_ind, y_ind))
                if self.is_in_map(map_x, map_y):
                    # odds ratio is updated according to the inverse sensor model
                    self.odds_ratios[x_detect, y_detect] *= self.p_occ / (1 - self.p_occ) * self.odds_ratio_hit

        self.seq += 1
        # to save time, only publish the map every 10 scans that we process
        if self.seq % 10 == 0:
            # make occupancy grid
            map = OccupancyGrid()
            map.header.seq = self.seq
            self.seq += 1
            map.header.stamp = msg.header.stamp
            map.header.frame_id = "map"  # the name of the coordinate frame of the map
            map.info.origin.position.x = self.origin[0]
            map.info.origin.position.y = self.origin[1]
            map.info.width = self.n
            map.info.height = self.n
            map.info.resolution = self.resolution
            map.data = [0] * self.n ** 2  # map.data stores the n by n grid in row-major order
            for i in range(self.n):
                for j in range(self.n):
                    idx = i + self.n * j  # this implements row major order
                    if self.odds_ratios[i, j] < 1 / 5.0:  # consider a cell free if odds ratio is low enough
                        map.data[idx] = 0
                    elif self.odds_ratios[i, j] > 5.0:  # consider a cell occupied if odds ratio is high enough
                        map.data[idx] = 100
                    else:  # otherwise cell is unknown
                        map.data[idx] = -1
            self.pub.publish(map)

        # create the image from the probabilities so we can visualize using opencv
        im = np.zeros((self.odds_ratios.shape[0], self.odds_ratios.shape[1], 3))
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if self.odds_ratios[i, j] < 1 / 5.0:
                    im[i, j, :] = 1.0
                elif self.odds_ratios[i, j] > 5.0:
                    im[i, j, :] = 0.0
                else:
                    im[i, j, :] = 0.5

        # compute the index of the odometry pose so we can mark it with a circle
        x_odom_index = int((self.odom_pose[0] - self.origin[0]) / self.resolution)
        y_odom_index = int((self.odom_pose[1] - self.origin[1]) / self.resolution)

        # computer the ball locations so we can mark with a colored circle
        #TODO Add stuff for each color so can do more than one at a time
        self.x_camera_red = x_odom_index + int(((-self.y_transform_red + self.odom_pose[0]) - self.origin[0]) * self.resolution)
        self.y_camera_red = y_odom_index + int(((-self.depth_red + self.odom_pose[1]) - self.origin[1]) * self.resolution)

        self.x_camera_blue = x_odom_index + int(((-self.y_transform_blue + self.odom_pose[0]) - self.origin[0]) * self.resolution)
        self.y_camera_blue = y_odom_index + int(((-self.depth_blue + self.odom_pose[1]) - self.origin[1]) * self.resolution)

        self.x_camera_green = x_odom_index + int(((-self.y_transform_green + self.odom_pose[0]) - self.origin[0]) * self.resolution)
        self.y_camera_green = y_odom_index + int(((-self.depth_green + self.odom_pose[1]) - self.origin[1]) * self.resolution)

        self.x_camera_yellow = x_odom_index + int(((-self.y_transform_yellow + self.odom_pose[0]) - self.origin[0]) * self.resolution)
        self.y_camera_yellow = y_odom_index + int(((-self.depth_yellow + self.odom_pose[1]) - self.origin[1]) * self.resolution)

        print self.x_camera_green
        print self.y_camera_green
        print self.y_transform_green

        # draw the circle
        cv2.circle(im, (y_odom_index, x_odom_index), 2, (255, 0, 0))
        # cv2.circle(im, (self.y_camera_red, self.x_camera_red), 2, self.red)
        # cv2.circle(im, (self.y_camera_yellow, self.x_camera_yellow), 2, self.yellow)
        cv2.circle(im, (self.y_camera_green, self.x_camera_green), 2, self.green)
        # cv2.circle(im, (self.y_camera_blue, self.x_camera_blue), 2, self.blue)
        # display the image resized
        cv2.imshow("map", cv2.resize(im, (500, 500)))
        cv2.waitKey(20)

    def coordinate_to_map_red(self, msg):
        x = msg.x
        y = msg.y
        r = msg.z

        self.depth_red = (r * self.depth_proportion + self.depth_intercept)
        #print depth
        self.y_transform_red = int(self.frame_height / 2 - y)
        self.x_transform_red = int(x - self.frame_width / 2)
        self.angle_diff_red = self.x_transform_red

    def coordinate_to_map_green(self, msg):
        print 'hey green!'
        x = msg.x
        y = msg.y
        r = msg.z

        self.depth_green = (r * self.depth_proportion + self.depth_intercept)
        print self.depth_green
        self.y_transform_green = int(self.frame_height / 2 - y)
        self.x_transform_green = int(x - self.frame_width / 2)
        self.angle_diff_green = self.x_transform_green

    def coordinate_to_map_blue(self, msg):
        x = msg.x
        y = msg.y
        r = msg.z

        self.depth_blue = (r * self.depth_proportion + self.depth_intercept)
        #print depth
        self.y_transform_blue = int(self.frame_height / 2 - y)
        self.x_transform_blue = int(x - self.frame_width / 2)
        self.angle_diff_blue = self.x_transform_blue

    def coordinate_to_map_yellow(self, msg):
        x = msg.x
        y = msg.y
        r = msg.z

        self.depth_yellow = (r * self.depth_proportion + self.depth_intercept)
        #print depth
        self.y_transform_yellow = int(self.frame_height / 2 - y)
        self.x_transform_yellow = int(x - self.frame_width / 2)
        self.angle_diff_yellow = self.x_transform_yellow

    @staticmethod
    def convert_pose_to_xy_and_theta(pose):
        """ Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple """
        orientation_tuple = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        angles = euler_from_quaternion(orientation_tuple)
        return pose.position.x, pose.position.y, angles[2]


class ImageConverter:
    def __init__(self):
        #TODO Adjust publishers/subscribers so that can publish more than one color at a time
        print("I'm initialized!")
        self.image_pub = rospy.Publisher("/processed_image", Image)
        self.ball_pub_red= rospy.Publisher("/ball_coords_red", Vector3)
        self.ball_pub_green= rospy.Publisher("/ball_coords_green", Vector3)
        self.ball_pub_blue= rospy.Publisher("/ball_coords_blue", Vector3)
        self.ball_pub_yellow= rospy.Publisher("/ball_coords_yellow", Vector3)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.ball_location_red = None
        self.ball_location_green = None
        self.ball_location_yellow = None
        self.ball_location_blue = None

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #Image Processing
        blur = cv2.medianBlur(cv_image, 7)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 5, 50)
        self.find_circles(edges, cv_image)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            if self.ball_location_yellow:
                self.ball_pub_yellow.publish(self.ball_location_yellow)
            if self.ball_location_red:
                self.ball_pub_red.publish(self.ball_location_red)
            if self.ball_location_blue:
                self.ball_pub_blue.publish(self.ball_location_blue)
            if self.ball_location_green:
                self.ball_pub_green.publish(self.ball_location_green)
            
        except CvBridgeError as e:
            print(e)

    def find_circles(self, img_src, img_out):
        """Finds and plots circles using Hough Circle detection."""
        circles = cv2.HoughCircles(img_src, cv2.cv.CV_HOUGH_GRADIENT, 1, img_src.shape[0] / 8, param1=10, param2=30,
                                   minRadius=40, maxRadius=100)
        hsv_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 200, 0])
        upper_red = np.array([20, 255, 255])
        mask_red = cv2.inRange(hsv_img, lower_red, upper_red)

        lower_blue = np.array([50, 200, 0])
        upper_blue = np.array([100, 255, 255])
        mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)

        lower_yellow = np.array([20, 25, 25])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        lower_green = np.array([60, 200, 10])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

        if circles is not None:
            for c in circles[0, :]:
                ROI_red = mask_red[c[1] - c[2]:c[1] + c[2], c[0] - c[2]:c[0] + c[2]]
                mean_red = cv2.mean(ROI_red)

                ROI_green = mask_green[c[1] - c[2]:c[1] + c[2], c[0] - c[2]:c[0] + c[2]]
                mean_green = cv2.mean(ROI_green)

                ROI_yellow = mask_yellow[c[1] - c[2]:c[1] + c[2], c[0] - c[2]:c[0] + c[2]]
                mean_yellow = cv2.mean(ROI_yellow)

                ROI_blue = mask_blue[c[1] - c[2]:c[1] + c[2], c[0] - c[2]:c[0] + c[2]]
                mean_blue = cv2.mean(ROI_blue)

                cv2.circle(img_out, (c[0], c[1]), c[2], (0, 255, 0), 1)
                # draw the center of the circle
                cv2.circle(img_out, (c[0], c[1]), 2, (0, 0, 255), 3)

                if mean_yellow[0] > 150:
                    #print mean
                    cv2.rectangle(img_out, (c[0] - c[2], c[1] + c[2]), (c[0] + c[2], c[1] - c[2]), (0, 255, 255), 2)
                    # draw the outer circle
                    cv2.circle(img_out, (c[0], c[1]), c[2], (0, 255, 255), 5)
                    # draw the center of the circle
                    cv2.circle(img_out, (c[0], c[1]), 2, (0, 255, 255), 3)
                    #print (c[0],c[1],c[2])
                    self.ball_location_yellow = Vector3(c[0], c[1], c[2])

                if mean_blue[0] > 50:
                    #print mean
                    cv2.rectangle(img_out, (c[0] - c[2], c[1] + c[2]), (c[0] + c[2], c[1] - c[2]), (255, 0, 0), 2)
                    # draw the outer circle
                    cv2.circle(img_out, (c[0], c[1]), c[2], (255, 0, 0), 5)
                    # draw the center of the circle
                    cv2.circle(img_out, (c[0], c[1]), 2, (255, 0, 0), 3)
                    #print (c[0],c[1],c[2])
                    self.ball_location_blue = Vector3(c[0], c[1], c[2])

                if mean_red[0] > 100:
                    #print mean
                    cv2.rectangle(img_out, (c[0] - c[2], c[1] + c[2]), (c[0] + c[2], c[1] - c[2]), (0, 0, 255), 2)
                    # draw the outer circle
                    cv2.circle(img_out, (c[0], c[1]), c[2], (0, 0, 255), 5)
                    # draw the center of the circle
                    cv2.circle(img_out, (c[0], c[1]), 2, (0, 0, 255), 3)
                    #print (c[0],c[1],c[2])
                    self.ball_location_red = Vector3(c[0], c[1], c[2])

                if mean_green[0] > 50:
                    #print mean
                    cv2.rectangle(img_out, (c[0] - c[2], c[1] + c[2]), (c[0] + c[2], c[1] - c[2]), (0, 255, 0), 2)
                    # draw the outer circle
                    cv2.circle(img_out, (c[0], c[1]), c[2], (0, 255, 0), 5)
                    # draw the center of the circle
                    cv2.circle(img_out, (c[0], c[1]), 2, (0, 255, 0), 3)
                    #print (c[0],c[1],c[2])
                    self.ball_location_green = Vector3(c[0], c[1], c[2])


class Controller:
    def __init__(self):
        #rospy.init_node('controller', anonymous=True)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.movement = self.follow_wall  # reassign to change behavior
        self.bridge = cv_bridge.CvBridge()
        self.running = False
        self.cmd_vel = Twist()
        signal.signal(signal.SIGINT, self.stop_neato)
        rospy.Subscriber("scan", LaserScan, self.laser_scan_received, queue_size=1)

        self.side = None
        self.lead_left_avg = 0
        self.lead_right_avg = 0
        self.trailing_left_avg = 0
        self.trailing_right_avg = 0

        self.lin_speed = 0.01
        self.spin_speed = 0.01

    def laser_scan_received(self, laser_scan_message):
        """Process laser scan points from LiDAR"""

        # Process some of the data
        lead_left_distance = []
        lead_right_distance = []
        trailing_left_distance = []
        trailing_right_distance = []
        for i in range(11):
            if 0 < laser_scan_message.ranges[i + 40] < 2:
                lead_left_distance.append(laser_scan_message.ranges[i + 40])
            if 0 < laser_scan_message.ranges[i + 130] < 2:
                trailing_left_distance.append(laser_scan_message.ranges[i + 130])
            if 0 < laser_scan_message.ranges[i + 310] < 2:
                lead_right_distance.append(laser_scan_message.ranges[i + 310])
            if 0 < laser_scan_message.ranges[i + 220] < 2:
                trailing_right_distance.append(laser_scan_message.ranges[i + 220])
        if len(lead_left_distance) + len(trailing_left_distance) > len(trailing_right_distance) + len(
                lead_right_distance):
            self.lead_left_avg = sum(lead_left_distance) / float(len(lead_left_distance) + 0.1)
            self.trailing_left_avg = sum(trailing_left_distance) / float(len(trailing_left_distance) + 0.1)
            self.side = 'left'
        else:
            self.lead_right_avg = sum(lead_right_distance) / float(len(lead_right_distance) + 0.1)
            self.trailing_right_avg = sum(trailing_right_distance) / float(len(trailing_right_distance) + 0.1)
            self.side = 'right'

    def follow_wall(self):
        linear_velocity = self.lin_speed

        if self.side == 'right':
            right_prop_dist = (self.lead_right_avg - self.trailing_right_avg)
            if self.lead_right_avg < 1.0 - 0.1 and self.lead_right_avg != 0:
                angular_velocity = self.lead_right_avg
                # print("move towards")
            elif self.lead_right_avg > 1.0 + 0.1:
                angular_velocity = -self.lead_right_avg
                # print("move away")
            else:
                if self.lead_right_avg - 0.1 < self.trailing_right_avg < 0.1 + self.lead_right_avg:
                    angular_velocity = 0.0
                    # print("Straight Ahead")
                else:
                    angular_velocity = right_prop_dist
                    # print("Maintain")

        elif self.side == 'left':
            # print("Left!")
            left_prop_dist = (self.lead_left_avg - self.trailing_left_avg)
            if self.lead_left_avg < 1.0 - 0.1 and self.lead_left_avg != 0:
                angular_velocity = -self.lead_left_avg
            elif self.lead_left_avg > 1.0 + 0.1:
                angular_velocity = self.lead_left_avg
            else:
                if self.lead_left_avg - 0.1 < self.trailing_left_avg < 0.1 + self.lead_left_avg:
                    angular_velocity = 0.0
                else:
                    angular_velocity = -left_prop_dist
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        # return Twist(Vector3(linear_velocity, 0.0, 0.0), Vector3(0.0, 0.0, angular_velocity))
        return Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0))

    def run(self):
        """Subscribe to the laser scan data and images."""
        self.running = True
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.running:
            self.cmd_vel = self.follow_wall()
            self.pub.publish(self.cmd_vel)
            rate.sleep()

    def stop_neato(self, signal, frame):
        """Stop the neato in case of signal interrupt."""
        self.running = False
        self.pub.publish(Twist())
        sys.exit(0)


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = ImageConverter()
    rescuebot = Controller()
    star_center = OccupancyGridMapper()
    try:
        rescuebot.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main(sys.argv)
