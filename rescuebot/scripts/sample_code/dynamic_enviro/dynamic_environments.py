#!/usr/bin/env python

#interfaces with ROS and Python
import rospy
#Getting topics from ROS
from std_msgs.msg import Header, String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion, Vector3
from nav_msgs.msg import OccupancyGrid
#Broadcaster/reciever information
import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
#Helper imports
import math
import time
import cv2 #openCV used to make window for publishing data
import numpy as np
from numpy.random import random_sample
from matplotlib.pyplot import imshow

class TransformHelpers:
	""" Helper functions for making transformations of data from the RunMapping
	world to the script and back.  Will only be useful for us if we add an autonomy
	function to the robot. Credits to Paul Ruvulo """

	@staticmethod
	def convert_pose_to_xy_and_theta(pose):
		""" Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple """
		orientation_tuple = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
		angles = euler_from_quaternion(orientation_tuple)
		return (pose.position.x, pose.position.y, angles[2])

class RunMapping:
	""" Creates a display for an Occupancy Field map
		Tracks the robot's position, can be expanded to track cluster positions
	"""

	def __init__(self):
		cv2.namedWindow("map")
		cv2.namedWindow("past_map")
		rospy.init_node("run_mapping")	
		#create map properties, helps to make ratio calcs
		self.origin = [-10,-10]
		self.seq = 0
		self.resolution = 0.1
		self.n = 200

		self.pose = []
		self.cluster_pose = []

		self.dyn_obs=[]
		self.rapid_appear = set()
		self.counter=0

		#Giving initial hypotheses to the system
		self.p_occ = 0.5*np.ones((self.n, self.n)) #50-50 chance of being occupied
		self.odds_ratio_hit = 3.0 #this is arbitrary, can re-assign
		self.odds_ratio_miss = 0.3 #this is arbitrary, can reassign

		#calculates odds based upon hit to miss, equal odds to all grid squares
		self.odds_ratios = (self.p_occ)/(1-self.p_occ)*np.ones((self.n, self.n))
		#calculate initial past odds_ratio
		self.past_odds_ratios = (self.p_occ)/(1-self.p_occ)*np.ones((self.n, self.n))

		#write laser pubs and subs
		rospy.Subscriber("scan", LaserScan, self.scan_received, queue_size=1)
		self.pub = rospy.Publisher("map", OccupancyGrid)

		#note - in case robot autonomy is added back in
		self.tf_listener = TransformListener()	
	

	def is_in_map(self, x, y):
		"Returns boolean of whether or not a point is within map boundaries"
		#return if x is less than the origin, or larger than the map, ditto for y
		return (x < self.origin[0] or x > self.origin[0] + self.n*self.resolution or y<self.origin[1] or y>self.origin[1] + self.n*self.resolution)

	def scan_received(self, msg):
		""" Returns an occupancy grid to publish data to map"""	
		if len(msg.ranges) != 360:
			return

		#make a pose stamp that relates to the odom of the robot
		p = PoseStamped(header=Header(stamp=msg.header.stamp,frame_id="base_link"), pose=Pose())
		self.odom_pose = self.tf_listener.transformPose("odom", p)
		# convert the odom pose to the tuple (x,y,theta)
		self.odom_pose = TransformHelpers.convert_pose_to_xy_and_theta(self.odom_pose.pose)

		#local variable used to take odds samples for future comparisons
		time_past = 0

		for degree in range(360):
			if msg.ranges[degree] > 0.0 and msg.ranges[degree] < 5.0:
				#gets the position of the laser data points
				data_x = self.odom_pose[0] + msg.ranges[degree]*math.cos(degree*math.pi/180.0 + self.odom_pose[2])
				data_y = self.odom_pose[1] + msg.ranges[degree]*math.sin(degree*math.pi/180+self.odom_pose[2])

				#maps laser data to a pixel in the map
				datax_pixel = int((data_x - self.origin[0])/self.resolution)
				datay_pixel = int((data_y - self.origin[1])/self.resolution)

				#maps the robot to a position
				robot = (data_x - self.odom_pose[0], data_y - self.odom_pose[1])

				#finds how far away the point is from the robot
				magnitude = math.sqrt(robot[0]**2 + robot[1]**2)

				#converts magnitude and robot position to pixels in the map
				n_steps = max([1, int(math.ceil(magnitude/self.resolution))])
				robot_step = (robot[0]/(n_steps-1), robot[1]/(n_steps-1))
				marked = set()

				for pixel in range(n_steps):
					curr_x = self.odom_pose[0] + pixel*robot_step[0]
					curr_y = self.odom_pose[1] + pixel*robot_step[1]
					if (self.is_in_map(curr_x, curr_y)):
						#make sure its in the map
						break

					x_ind = int((curr_x - self.origin[0])/self.resolution)
					y_ind = int((curr_y - self.origin[1])/self.resolution)
					if x_ind == datax_pixel and y_ind==datay_pixel and self.odds_ratios[datax_pixel, datay_pixel] >= 1/60.0:
						#set odds ratio equal to past odds ratio
						if time_past % 5 == 0:
							self.past_odds_ratios[datax_pixel, datay_pixel]=self.odds_ratios[datax_pixel, datay_pixel]
							time_past += 1
						self.odds_ratios[datax_pixel, datay_pixel] *= self.p_occ[datax_pixel, datay_pixel]/(1-self.p_occ[datax_pixel, datay_pixel]) * self.odds_ratio_hit
					if not((x_ind, y_ind) in marked) and self.odds_ratios[x_ind, y_ind] >= 1/60.0:
						#If point isn't marked, update the odds of missing and add to the map
						if time_past % 5 == 0:
							self.past_odds_ratios[x_ind, y_ind]=self.odds_ratios[x_ind, y_ind]
							time_past += 1
						self.odds_ratios[x_ind, y_ind] *= self.p_occ[x_ind, y_ind] / (1-self.p_occ[x_ind, y_ind]) * self.odds_ratio_miss
						marked.add((x_ind, y_ind))
				if not(self.is_in_map(data_x, data_y)) and self.odds_ratios[data_x, datay_pixel] >= 1/60.0:
					#if it is not in the map, update the odds of hitting it
					if time_past % 5 == 0:
						self.past_odds_ratios[datax_pixel, datay_pixel]=self.odds_ratios[datax_pixel, datay_pixel]
						time_past += 1
					self.odds_ratios[datax_pixel, datay_pixel] *= self.p_occ[datax_pixel, datay_pixel]/(1-self.p_occ[datax_pixel, datay_pixel]) * self.odds_ratio_hit


		self.seq += 1
		if self.seq % 10 == 0:
			map = OccupancyGrid() #this is a nav msg class
			map.header.seq = self.seq
			map.header.stamp = msg.header.stamp
			map.header.frame_id = "map"
			map.header.frame_id = "past_map"
			map.info.origin.position.x = self.origin[0]
			map.info.origin.position.y = self.origin[1]
			map.info.width = self.n
			map.info.height = self.n
			map.info.resolution = self.resolution
			map.data = [0]*self.n**2 #the zero is a formatter, not a multiple of 0
			for i in range(self.n):
				#this is giving us the i,j grid square, occupancy grid
				for j in range(self.n):
					idx = i+self.n*j #makes horizontal rows (i is x, j is y)
					if self.odds_ratios[i,j] < 1/50.0:
						map.data[idx] = 0 #makes the gray
					elif self.odds_ratios[i,j] >= 1/50.0 < 4.0/5.0:
						map.data[idx] = 25
					elif self.odds_ratios[i,j] > 50.0:
						map.data[idx] = 100 #makes the black walls
					else:
						map.data[idx] = -1 #makes unknown
			self.pub.publish(map)

		image = np.zeros((self.odds_ratios.shape[0], self.odds_ratios.shape[1],3))
		image2 = np.zeros((self.odds_ratios.shape[0], self.odds_ratios.shape[1],3))

		self.counter+=1

		#get robots position
		x_odom_index = int((self.odom_pose[0] - self.origin[0])/self.resolution)
		y_odom_index = int((self.odom_pose[1] - self.origin[1])/self.resolution)

		#remember robots position over time
		self.pose.append((x_odom_index, y_odom_index))

		#.shape() comes from being related to the np class
		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				#the thing that just rapidly appeared, disappeared!
				delta = (self.odds_ratios[i,j]-self.past_odds_ratios[i,j])
				if (delta < 0.0) and (i,j) in self.rapid_appear:
					self.dyn_obs.append((i,j,self.counter))
				#whoa buddy, a thing just appeared!
				if delta > 1000.0 and (i,j) not in self.rapid_appear:
					self.rapid_appear.add((i,j))
				#Map the world with particular colors
				if self.odds_ratios[i,j] < 1/50.0:
					image[i,j,:] = 1.0 #makes open space
				elif self.odds_ratios[i,j] >= 1/50.0 and self.odds_ratios[i,j] <4/5.0:
					image[i,j,:] = (0, 255, 0) #uncertainty points
				elif self.odds_ratios[i,j] > 50.0:
					image[i,j,:] = (0, 0, 255) #makes walls
				else:
					image[i,j,:] = 0.5 #not read
					
		#if a dynamic obstacle is recorded, let's see it!
		if len(self.dyn_obs)>0:
			for point in self.dyn_obs:
				if (self.counter-point[2])<=10:
					image2[point[0],point[1]] = (255,0,255) #makes old/dynamic shapes on other map
		#also, we want to map the robot's trail
		for point in self.pose:
			image[point[0], point[1]] = (255, 0, 0)

		#draw it!
		cv2.circle(image,(y_odom_index, x_odom_index), 2,(255,0,0))
		cv2.imshow("map", cv2.resize(image,(500,500)))
		cv2.imshow("past_map", cv2.resize(image2,(500,500)))
		cv2.waitKey(20) #effectively a delay


	def run(self):
		r = rospy.Rate(10)
		while not(rospy.is_shutdown()):
			r.sleep()


if __name__ == '__main__':
	try:
		n = RunMapping()
		n.run()
	except rospy.ROSInterruptException: pass