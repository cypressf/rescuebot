#!/usr/bin/env python
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Pose
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
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
        #rospy.init_node("occupancy_grid_mapper")
        self.origin = [-10, -10]
        self.seq = 0
        self.resolution = .1
        self.n = 200
        self.p_occ = .5
        self.odds_ratio_hit = 3.0
        self.odds_ratio_miss = .2
        self.odds_ratios = (self.p_occ)/(1-self.p_occ)*np.ones((self.n,self.n))
        rospy.Subscriber("scan", LaserScan, self.process_scan, queue_size=1)
        self.pub = rospy.Publisher("map",OccupancyGrid)
        self.tf_listener = TransformListener()

    def is_in_map(self,x_ind,y_ind):
        """ Return whether or not the given point is within the map boundaries """
        return not(x_ind < self.origin[0] or
                   x_ind > self.origin[0] + self.n*self.resolution or
                   y_ind < self.origin[1] or
                   y_ind > self.origin[1] + self.n*self.resolution)

    def process_scan(self, msg):
        """ Callback function for the laser scan messages """
        if len(msg.ranges) != 360:
            # throw out scans that don't have all the laser data
            return
        # get pose according to the odometry
        p = PoseStamped(header=Header(stamp=msg.header.stamp,frame_id="base_link"), pose=Pose())
        self.odom_pose = self.tf_listener.transformPose("odom", p)
        # convert the odom pose to the tuple (x,y,theta)
        self.odom_pose = OccupancyGridMapper.convert_pose_to_xy_and_theta(self.odom_pose.pose)
        for i in range(360):
            if msg.ranges[i] > 0.0 and msg.ranges[i] < 5.0:
                # TODO: draw a picture for this to understand it better
                map_x = self.odom_pose[0] + msg.ranges[i]*cos(i*pi/180.0+self.odom_pose[2])
                map_y = self.odom_pose[1] + msg.ranges[i]*sin(i*pi/180.0+self.odom_pose[2])

                x_detect = int((map_x - self.origin[0])/self.resolution)
                y_detect = int((map_y - self.origin[1])/self.resolution)

                u = (map_x - self.odom_pose[0], map_y - self.odom_pose[1])
                magnitude = sqrt(u[0]**2 + u[1]**2)
                n_steps = max([1,int(ceil(magnitude/self.resolution))])
                u_step = (u[0]/(n_steps-1), u[1]/(n_steps-1))
                marked = set()
                for i in range(n_steps):
                    curr_x = self.odom_pose[0] + i*u_step[0]
                    curr_y = self.odom_pose[1] + i*u_step[1]
                    if not(self.is_in_map(curr_x,curr_y)):
                        break

                    x_ind = int((curr_x - self.origin[0])/self.resolution)
                    y_ind = int((curr_y - self.origin[1])/self.resolution)
                    if x_ind == x_detect and y_ind == y_detect:
                        break
                    if not((x_ind,y_ind) in marked):
                        # odds ratio is updated according to the inverse sensor model
                        self.odds_ratios[x_ind,y_ind] *= self.p_occ / (1-self.p_occ) * self.odds_ratio_miss
                        marked.add((x_ind,y_ind))
                if self.is_in_map(map_x, map_y):
                    # odds ratio is updated according to the inverse sensor model
                    self.odds_ratios[x_detect,y_detect] *= self.p_occ / (1-self.p_occ) * self.odds_ratio_hit

        self.seq += 1
        # to save time, only publish the map every 10 scans that we process
        if self.seq % 10 == 0:
            # make occupancy grid
            map = OccupancyGrid()
            map.header.seq = self.seq
            self.seq += 1
            map.header.stamp = msg.header.stamp
            map.header.frame_id = "map"                     # the name of the coordinate frame of the map
            map.info.origin.position.x = self.origin[0]
            map.info.origin.position.y = self.origin[1]
            map.info.width = self.n
            map.info.height = self.n
            map.info.resolution = self.resolution
            map.data = [0]*self.n**2                        # map.data stores the n by n grid in row-major order
            for i in range(self.n):
                for j in range(self.n):
                    idx = i+self.n*j                        # this implements row major order
                    if self.odds_ratios[i,j] < 1/5.0:       # consider a cell free if odds ratio is low enough
                        map.data[idx] = 0
                    elif self.odds_ratios[i,j] > 5.0:       # consider a cell occupied if odds ratio is high enough
                        map.data[idx] = 100
                    else:                                   # otherwise cell is unknown
                        map.data[idx] = -1
            self.pub.publish(map)

        # create the image from the probabilities so we can visualize using opencv
        im = np.zeros((self.odds_ratios.shape[0],self.odds_ratios.shape[1],3))
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if self.odds_ratios[i,j] < 1/5.0:
                    im[i,j,:] = 1.0
                elif self.odds_ratios[i,j] > 5.0:
                    im[i,j,:] = 0.0
                else:
                    im[i,j,:] = 0.5

        # compute the index of the odometry pose so we can mark it with a circle
        x_odom_index = int((self.odom_pose[0] - self.origin[0])/self.resolution)
        y_odom_index = int((self.odom_pose[1] - self.origin[1])/self.resolution)

        # draw the circle
        cv2.circle(im,(y_odom_index, x_odom_index),2,(255,0,0))
        # display the image resized
        cv2.imshow("map",cv2.resize(im,(500,500)))
        cv2.waitKey(20)

    @staticmethod
    def convert_pose_to_xy_and_theta(pose):
        """ Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple """
        orientation_tuple = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        angles = euler_from_quaternion(orientation_tuple)
        return (pose.position.x, pose.position.y, angles[2])

    def run(self):
        r = rospy.Rate(10)
        while not(rospy.is_shutdown()):
            r.sleep()