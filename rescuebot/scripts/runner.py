#!/usr/bin/env python
from __future__ import division
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Pose
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from points import LaserPoint, ranges_to_points, filter_points
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
import time
import dynamic_reconfigure.client


class OccupancyGridMapper:
    """ Implements simple occupancy grid mapping """

    def __init__(self):
        cv2.namedWindow("map")
        print 'Running'

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
        #TODO Track and relate the robot's angle pose for accuracy
        if self.depth_red != 0:
            self.x_camera_red = x_odom_index + int(((-self.y_transform_red * cos(self.odom_pose[2]) + self.odom_pose[0]) - self.origin[0]) * self.resolution)
            self.y_camera_red = y_odom_index + int(((-self.depth_red * sin( self.odom_pose[2])+ self.odom_pose[1]) - self.origin[1]) * self.resolution)
            cv2.circle(im, (self.y_camera_red, self.x_camera_red), 2, self.red)

        if self.depth_blue != 0:
            self.x_camera_blue = x_odom_index + int(((-self.y_transform_blue + self.odom_pose[0]) - self.origin[0]) * self.resolution)
            self.y_camera_blue = y_odom_index + int(((-self.depth_blue + self.odom_pose[1]) - self.origin[1]) * self.resolution)
            cv2.circle(im, (self.y_camera_blue, self.x_camera_blue), 2, self.blue)

        if self.depth_green != 0:
            self.x_camera_green = x_odom_index + int(((-self.y_transform_green + self.odom_pose[0]) - self.origin[0]) * self.resolution)
            self.y_camera_green = y_odom_index + int(((-self.depth_green + self.odom_pose[1]) - self.origin[1]) * self.resolution)
            cv2.circle(im, (self.y_camera_green, self.x_camera_green), 2, self.green)

        if self.depth_yellow != 0:
            self.x_camera_yellow = x_odom_index + int(((-self.y_transform_yellow + self.odom_pose[0]) - self.origin[0]) * self.resolution)
            self.y_camera_yellow = y_odom_index + int(((-self.depth_yellow + self.odom_pose[1]) - self.origin[1]) * self.resolution)
            cv2.circle(im, (self.y_camera_yellow, self.x_camera_yellow), 2, self.yellow)

        # draw the robot
        cv2.circle(im, (y_odom_index, x_odom_index), 2, (255, 0, 0))
        
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
        print x
        print y
        print r
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
        print("Image converter initialized!")

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
        self.bridge = cv_bridge.CvBridge()
        self.running = False
        self.cmd_vel = Twist()
        signal.signal(signal.SIGINT, self.stop_neato)
        rospy.Subscriber("scan", LaserScan, self.laser_scan_received, queue_size=1)

        self.MAX_LINEAR_SPEED = 0.04
        self.MAX_ANGULAR_SPEED = .28
        self.DANGER_ZONE_LENGTH = 1.0
        self.DANGER_ZONE_WIDTH = 0.5
        self.DANGER_POINTS_MULTIPLIER = 1/50.0
        self.WALL_FOLLOW_DISTANCE = .5
        self.ROOM_CENTER_CUTOFF = 0.5
        self.room_center_number_points = 60
        self.angle_wall_tolerance = 0.1
        self.wall_distance_constant = 0.5

        self.side = None
        self.lead_left_avg = 0
        self.lead_right_avg = 0
        self.trailing_left_avg = 0
        self.trailing_right_avg = 0

        self.get_cmd_vel = self.follow_wall

        self.points = []
        self.filtered_points = []

    def laser_scan_received(self, laser_scan_message):
        """Process laser scan points from LiDAR"""

        self.points = ranges_to_points(laser_scan_message.ranges)
        self.filtered_points = filter_points(self.points)

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
        # if self.get_danger_points():
        #     self.get_cmd_vel = self.obstacle_avoid
        #     return self.obstacle_avoid()

        linear_velocity = self.MAX_LINEAR_SPEED

        if self.side == 'right':
            rospy.loginfo('right')
            angular_velocity = self.follow_wall_helper(self.lead_right_avg, self.trailing_right_avg, 'right')

        elif self.side == 'left':
            rospy.loginfo('left')
            angular_velocity = -self.follow_wall_helper(self.lead_left_avg, self.trailing_left_avg, 'left')

        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        return Twist(Vector3(linear_velocity, 0.0, 0.0), Vector3(0.0, 0.0, angular_velocity))

    def follow_wall_helper(self, lead_avg, trailing_avg, wall_direction):
        point_to_wall = self.get_point_to_wall()
        if point_to_wall:
            angle_to_wall = point_to_wall.angle_radians / (2 * pi)
            distance_to_wall = point_to_wall.length
        else:
            return 0

        if lead_avg + trailing_avg == 0:
            rospy.loginfo("no data")
            return 0

        if not self.WALL_FOLLOW_DISTANCE - 0.1 < lead_avg < self.WALL_FOLLOW_DISTANCE + 0.1:
            goal_angle = (self.WALL_FOLLOW_DISTANCE - distance_to_wall) * self.wall_distance_constant + 1/4
            if goal_angle < 1/8:
                goal_angle = 1/8
            if goal_angle > 3/8:
                goal_angle = 3/8
            if wall_direction == "right":
                goal_angle = 1 - goal_angle

            angular_velocity = self.MAX_ANGULAR_SPEED * (angle_to_wall - goal_angle) / (3/8)
        else:
            proportional_distance = (lead_avg - trailing_avg) / (lead_avg + trailing_avg)
            angular_velocity = -proportional_distance
            rospy.loginfo("correct distance: {:.4f} lead: {:.4f}, trailing: {:.4f}".format(angular_velocity, lead_avg, trailing_avg))
        if np.abs(angular_velocity) > self.MAX_ANGULAR_SPEED:
            angular_velocity = self.MAX_ANGULAR_SPEED * np.sign(angular_velocity)
        return angular_velocity

    def get_point_to_wall(self):
        """If we're close to a wall, return the point closest to us on the wall."""

        closest_point = min(self.filtered_points)
        theta = closest_point.angle_degrees

        # Constructs a second line 5 degrees off, checks to see if it forms proper right triangle
        dist_hyp = self.points[(theta+5)%360].length
        dist_opp = np.tan(np.radians(5)/closest_point.length)
        if abs(np.sqrt(closest_point.length**2 + dist_opp**2) - dist_hyp) <= self.angle_wall_tolerance:
            rospy.loginfo('Close enough -- wall detected')
            return closest_point
        else:
            rospy.loginfo('False alarm -- no wall detected')
            return False


    def is_in_danger_zone(self, point):
        a = self.DANGER_ZONE_LENGTH * sin(point.angle_radians)
        b = self.DANGER_ZONE_WIDTH * cos(point.angle_radians)
        max_radius = (self.DANGER_ZONE_LENGTH * self.DANGER_ZONE_WIDTH) / sqrt(a**2 + b**2)
        return point.length < max_radius and point.is_in_front()

    def get_danger_points(self):
        return [point for point in self.filtered_points if self.is_in_danger_zone(point)]

    def room_center(self):
        """
        Go to the center of the room.
        """
        std_dev = np.std([point.length for point in self.filtered_points])
        if std_dev < self.ROOM_CENTER_CUTOFF:
            self.get_cmd_vel = self.start_360()
            return self.start_360()
        closest_points = sorted(self.filtered_points)[:self.room_center_number_points]
        angles = [point.angle_radians for point in closest_points]
        imaginary_numbers = [np.exp(angle*1j) for angle in angles]
        angle_mean = np.angle(np.mean(imaginary_numbers))
        if angle_mean < 0:
            angle_mean += 2*pi

        angle = angle_mean / (2 * pi)
        if angle < 1/2:
            linear_velocity = np.interp(angle, [0, 1/2], [-self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED])
        else:
            linear_velocity = np.interp(angle, [1/2, 1], [self.MAX_LINEAR_SPEED, -self.MAX_LINEAR_SPEED])

        if 1/4 < angle < 3/4:
            angular_velocity = np.interp(angle, [1/4, 3/4], [-self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED])
        elif 0 <= angle <= 1/4:
            angular_velocity = np.interp(angle, [0, 1/4], [0, self.MAX_ANGULAR_SPEED])
        else:
            angular_velocity = np.interp(angle, [3/4, 1], [-self.MAX_ANGULAR_SPEED, 0])

        cmd_vel = Twist()
        cmd_vel.angular.z = angular_velocity
        cmd_vel.linear.x = linear_velocity
        rospy.loginfo("wall angle: {:.4f} -> linear: {:.4f}, angular: {:.4f}. std_dev: {:.3f}".format(angle, linear_velocity, angular_velocity, std_dev))
        return cmd_vel

    def start_360(self):
        self.time = 0
        self.get_cmd_vel = self.do_360
        return self.do_360()

    def do_360(self):
        """
        Turn around a full 360 degrees, then wall follow.
        """
        self.time+=1
        if self.time > 130 / self.MAX_ANGULAR_SPEED:
            self.get_cmd_vel = self.follow_wall
            return self.follow_wall()
        t = Twist()
        t.angular.z = self.MAX_ANGULAR_SPEED
        return t

    def obstacle_avoid(self):
        danger_points = self.get_danger_points()
        left_danger_points = [point for point in danger_points if point.is_in_front_left()]
        right_danger_points = [point for point in danger_points if point.is_in_front_right()]

        if not danger_points:
            self.get_cmd_vel = self.follow_wall
            return self.follow_wall()

        if len(left_danger_points) < len(right_danger_points):
            return self.turn_left(len(right_danger_points))
        else:
            return self.turn_right(len(left_danger_points))

    def turn_left(self, num_danger_points):
        angular_velocity = Vector3(z=num_danger_points * self.DANGER_POINTS_MULTIPLIER * self.MAX_ANGULAR_SPEED)
        linear_velocity = Vector3(x=self.MAX_LINEAR_SPEED * (1 - num_danger_points * self.DANGER_POINTS_MULTIPLIER))
        return Twist(angular=angular_velocity, linear=linear_velocity)

    def turn_right(self, num_danger_points):
        angular_velocity = Vector3(z=num_danger_points * self.DANGER_POINTS_MULTIPLIER * -self.MAX_ANGULAR_SPEED)
        linear_velocity = Vector3(x=self.MAX_LINEAR_SPEED * (1 - num_danger_points * self.DANGER_POINTS_MULTIPLIER))
        return Twist(angular=angular_velocity, linear=linear_velocity)

    def run(self):
        """Subscribe to the laser scan data and images."""
        self.running = True
        dynamic_reconfigure.client.Client("dynamic_reconfigure_server", timeout=5, config_callback=self.dynamic_reconfigure_callback)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.running:
            cmd_vel = self.get_cmd_vel()
            self.pub.publish(cmd_vel)
            rate.sleep()

    def stop_neato(self, signal, frame):
        """Stop the neato in case of signal interrupt."""
        self.running = False
        self.pub.publish(Twist())
        sys.exit(0)

    def dynamic_reconfigure_callback(self, config):
        self.MAX_LINEAR_SPEED = config["max_linear_speed"]
        self.MAX_ANGULAR_SPEED = config["max_angular_speed"]
        self.DANGER_ZONE_LENGTH = config["danger_zone_length"]
        self.DANGER_ZONE_WIDTH = config["danger_zone_width"]
        self.DANGER_POINTS_MULTIPLIER = config["danger_points_multiplier"]
        self.WALL_FOLLOW_DISTANCE = config["wall_follow_distance"]
        self.ROOM_CENTER_CUTOFF = config["room_center_cutoff"]
        self.room_center_number_points = config["room_center_number_points"]
        rospy.loginfo(config)

def main(args):
    rospy.init_node('image_converter', anonymous=True)
    # ic = ImageConverter()
    rescuebot = Controller()
    # star_center = OccupancyGridMapper()
    time.sleep(2)
    try:
        rescuebot.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main(sys.argv)
