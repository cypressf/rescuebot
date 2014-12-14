#!/usr/bin/env python
from __future__ import division
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Pose
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from points import LaserPoint, ranges_to_points
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


MAX_LINEAR_SPEED = 0.14
MAX_ANGULAR_SPEED = .28
DANGER_ZONE_LENGTH = 1.0
DANGER_ZONE_WIDTH = 0.5
DANGER_POINTS_MULTIPLIER = 1/50.0
WALL_FOLLOW_DISTANCE = .5
ROOM_CENTER_CUTOFF = 0.5

class OccupancyGridMapper:
    """ Implements simple occupancy grid mapping """

    def __init__(self):
        cv2.namedWindow("map")
        #rospy.init_node("occupancy_grid_mapper")
        self.origin = [-10, -10]
        self.seq = 0
        self.resolution = .1
        self.n = 200
        self.p_occ = .5
        self.odds_ratio_hit = 3.0
        self.odds_ratio_miss = .2
        self.odds_ratios = self.p_occ / (1 - self.p_occ) * np.ones((self.n, self.n))
        rospy.Subscriber("scan", LaserScan, self.process_scan, queue_size=1)
        self.pub = rospy.Publisher("map", OccupancyGrid)
        self.tf_listener = TransformListener()
        self.move_sub = rospy.Subscriber('ball_coords', Vector3, self.coordinate_to_map)
        self.color_sub = rospy.Subscriber('color', Vector3, self.process_color, queue_size=1)
        self.frame_height = 480
        self.frame_width = 640
        self.depth = 0
        #print depth
        self.y_transform = 0
        self.x_transform = 0
        self.angle_diff = 0
        self.color = (0, 0, 0)

    def is_in_map(self, x_ind, y_ind):
        """ Return whether or not the given point is within the map boundaries """
        return not (x_ind < self.origin[0] or
                    x_ind > self.origin[0] + self.n * self.resolution or
                    y_ind < self.origin[1] or
                    y_ind > self.origin[1] + self.n * self.resolution)

    def process_color(self, msg):
        self.color = (int(msg.x), int(msg.y), int(msg.z))

    def process_scan(self, msg):
        """ Callback function for the laser scan messages """
        if len(msg.ranges) != 360:
            # throw out scans that don't have all the laser data
            return
        # get pose according to the odometry
        p = PoseStamped(header=Header(stamp=msg.header.stamp, frame_id="base_link"), pose=Pose())
        self.odom_pose = self.tf_listener.transformPose("odom", p)
        # convert the odom pose to the tuple (x,y,theta)
        self.odom_pose = OccupancyGridMapper.convert_pose_to_xy_and_theta(self.odom_pose.pose)
        for i in range(360):
            if 0.0 < msg.ranges[i] < 5.0:
                map_x = self.odom_pose[0] + msg.ranges[i] * cos(i * pi / 180.0 + self.odom_pose[2])
                map_y = self.odom_pose[1] + msg.ranges[i] * sin(i * pi / 180.0 + self.odom_pose[2])

                x_detect = int((map_x - self.origin[0]) / self.resolution)
                y_detect = int((map_y - self.origin[1]) / self.resolution)

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

        x_camera = x_odom_index + int(((-self.x_transform + self.odom_pose[0]) - self.origin[0]) * self.resolution)
        y_camera = y_odom_index + int(((-self.depth + self.odom_pose[1]) - self.origin[1]) * self.resolution)

        print(x_camera)
        print(y_camera)

        # draw the circle
        cv2.circle(im, (y_odom_index, x_odom_index), 2, (255, 0, 0))
        cv2.circle(im, (y_camera, x_camera), 2, self.color)
        # display the image resized
        cv2.imshow("map", cv2.resize(im, (500, 500)))
        cv2.waitKey(20)

    def coordinate_to_map(self, msg):
        x = msg.x
        y = msg.y
        r = msg.z

        depth_proportion = -.025
        depth_intercept = 1.35
        self.depth = int(r * depth_proportion + depth_intercept)
        #print depth
        self.y_transform = int(self.frame_height / 2 - y)
        self.x_transform = int(x - self.frame_width / 2)
        self.angle_diff = self.x_transform

    @staticmethod
    def convert_pose_to_xy_and_theta(pose):
        """ Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple """
        orientation_tuple = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        angles = euler_from_quaternion(orientation_tuple)
        return pose.position.x, pose.position.y, angles[2]


class ImageConverter:
    def __init__(self):
        print("I'm initialized!")
        self.image_pub = rospy.Publisher("/processed_image", Image)
        self.ball_pub = rospy.Publisher("/ball_coords", Vector3)
        self.color_pub = rospy.Publisher("/color", Vector3)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.ball_location = None
        self.color = None

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
            if self.ball_location:
                self.ball_pub.publish(self.ball_location)
            if self.color:
                self.color_pub.publish(self.color)
        except CvBridgeError as e:
            print(e)

    def find_circles(self, img_src, img_out):
        """Finds and plots circles using Hough Circle detection."""
        circles = cv2.HoughCircles(img_src, cv2.cv.CV_HOUGH_GRADIENT, 1, img_src.shape[0] / 8, param1=10, param2=30,
                                   minRadius=40, maxRadius=75)
        hsv_img = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)

        lower_red = np.array([150, 55, 55])
        upper_red = np.array([200, 255, 255])
        mask_red = cv2.inRange(hsv_img, lower_red, upper_red)

        lower_blue = np.array([100, 55, 55])
        upper_blue = np.array([150, 255, 255])
        mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)

        lower_yellow = np.array([25, 25, 25])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        lower_green = np.array([90, 55, 55])
        upper_green = np.array([130, 255, 255])
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
                    self.ball_location = Vector3(c[0], c[1], c[2])
                    self.color = Vector3(0, 255, 255)

                if mean_blue[0] > 50:
                    #print mean
                    cv2.rectangle(img_out, (c[0] - c[2], c[1] + c[2]), (c[0] + c[2], c[1] - c[2]), (255, 0, 0), 2)
                    # draw the outer circle
                    cv2.circle(img_out, (c[0], c[1]), c[2], (255, 0, 0), 5)
                    # draw the center of the circle
                    cv2.circle(img_out, (c[0], c[1]), 2, (255, 0, 0), 3)
                    #print (c[0],c[1],c[2])
                    self.ball_location = Vector3(c[0], c[1], c[2])
                    self.color = Vector3(255, 0, 0)

                if mean_red[0] > 100:
                    #print mean
                    cv2.rectangle(img_out, (c[0] - c[2], c[1] + c[2]), (c[0] + c[2], c[1] - c[2]), (0, 0, 255), 2)
                    # draw the outer circle
                    cv2.circle(img_out, (c[0], c[1]), c[2], (0, 0, 255), 5)
                    # draw the center of the circle
                    cv2.circle(img_out, (c[0], c[1]), 2, (0, 0, 255), 3)
                    #print (c[0],c[1],c[2])
                    self.ball_location = Vector3(c[0], c[1], c[2])
                    self.color = Vector3(0, 0, 255)

                if mean_green[0] > 50:
                    #print mean
                    cv2.rectangle(img_out, (c[0] - c[2], c[1] + c[2]), (c[0] + c[2], c[1] - c[2]), (0, 255, 0), 2)
                    # draw the outer circle
                    cv2.circle(img_out, (c[0], c[1]), c[2], (0, 255, 0), 5)
                    # draw the center of the circle
                    cv2.circle(img_out, (c[0], c[1]), 2, (0, 255, 0), 3)
                    #print (c[0],c[1],c[2])
                    self.ball_location = Vector3(c[0], c[1], c[2])
                    self.color = Vector3(0, 255, 0)


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

        self.get_cmd_vel = self.room_center

        self.points = []

    def laser_scan_received(self, laser_scan_message):
        """Process laser scan points from LiDAR"""

        self.points = ranges_to_points(laser_scan_message.ranges)

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
        if self.get_danger_points():
            self.get_cmd_vel = self.obstacle_avoid
            return self.obstacle_avoid()

        linear_velocity = MAX_LINEAR_SPEED

        if self.side == 'right':
            right_prop_dist = (self.lead_right_avg - self.trailing_right_avg)
            if self.lead_right_avg < WALL_FOLLOW_DISTANCE - 0.1 and self.lead_right_avg != 0:
                angular_velocity = self.lead_right_avg
                # print("move towards")
            elif self.lead_right_avg > WALL_FOLLOW_DISTANCE + 0.1:
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
            if self.lead_left_avg < WALL_FOLLOW_DISTANCE - 0.1 and self.lead_left_avg != 0:
                angular_velocity = -self.lead_left_avg
            elif self.lead_left_avg > WALL_FOLLOW_DISTANCE + 0.1:
                angular_velocity = self.lead_left_avg
            else:
                if self.lead_left_avg - 0.1 < self.trailing_left_avg < 0.1 + self.lead_left_avg:
                    angular_velocity = 0.0
                else:
                    angular_velocity = -left_prop_dist
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        return Twist(Vector3(linear_velocity, 0.0, 0.0), Vector3(0.0, 0.0, angular_velocity))
        # return Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0))

    def is_in_danger_zone(self, point):
        a = DANGER_ZONE_LENGTH * sin(point.angle_radians)
        b = DANGER_ZONE_WIDTH * cos(point.angle_radians)
        max_radius = (DANGER_ZONE_LENGTH * DANGER_ZONE_WIDTH) / sqrt(a**2 + b**2)
        return point.length < max_radius and point.is_in_front()

    def get_danger_points(self):
        return [point for point in self.points if self.is_in_danger_zone(point)]

    def room_center(self):
        """
        Go to the center of the room.
        """
        std_dev = np.std([point.length for point in self.points])
        # rospy.loginfo(std_dev)
        if std_dev < ROOM_CENTER_CUTOFF:
            self.get_cmd_vel = self.start_360()
            return self.start_360()
        closest_points = sorted(self.points)[:60]
        angles = [point.angle_radians for point in closest_points]
        imaginary_numbers = [np.exp(angle*1j) for angle in angles]
        angle_mean = np.angle(np.mean(imaginary_numbers))
        if angle_mean < 0:
            angle_mean += 2*pi

        angle = angle_mean / (2 * pi)
        if angle < 1/2:
            linear_velocity = np.interp(angle, [0, 1/2], [-MAX_LINEAR_SPEED, MAX_LINEAR_SPEED])
        else:
            linear_velocity = np.interp(angle, [1/2, 1], [MAX_LINEAR_SPEED, -MAX_LINEAR_SPEED])

        if 1/4 < angle < 3/4:
            angular_velocity = np.interp(angle, [1/4, 3/4], [-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED])
        elif 0 <= angle <= 1/4:
            angular_velocity = np.interp(angle, [0, 1/4], [0, MAX_ANGULAR_SPEED])
        else:
            angular_velocity = np.interp(angle, [3/4, 1], [-MAX_ANGULAR_SPEED, 0])

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
        if self.time > 130 / MAX_ANGULAR_SPEED:
            self.get_cmd_vel = self.follow_wall
            return self.follow_wall()
        t = Twist()
        t.angular.z = MAX_ANGULAR_SPEED
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
        angular_velocity = Vector3(z=num_danger_points * DANGER_POINTS_MULTIPLIER * MAX_ANGULAR_SPEED)
        linear_velocity = Vector3(x=MAX_LINEAR_SPEED * (1 - num_danger_points * DANGER_POINTS_MULTIPLIER))
        return Twist(angular=angular_velocity, linear=linear_velocity)

    def turn_right(self, num_danger_points):
        angular_velocity = Vector3(z=num_danger_points * DANGER_POINTS_MULTIPLIER * -MAX_ANGULAR_SPEED)
        linear_velocity = Vector3(x=MAX_LINEAR_SPEED * (1 - num_danger_points * DANGER_POINTS_MULTIPLIER))
        return Twist(angular=angular_velocity, linear=linear_velocity)

    def run(self):
        """Subscribe to the laser scan data and images."""
        self.running = True
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.running:
            self.pub.publish(self.get_cmd_vel())
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
    # star_center = OccupancyGridMapper()
    try:
        rescuebot.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main(sys.argv)
