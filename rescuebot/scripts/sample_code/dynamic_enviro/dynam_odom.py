#!/usr/bin/env python

"""
October 3, 2014 - Adjustments were made to Paul's particle filter code 
to gather all the statements which interact with gazebo, rviz, or 
various rostopics.  The rest is a pseudo code structure.

October 4, 2014 - C+V: Finished formatting base code form Paul's inclass example (with some minor adjustments and renamings, and from pf_code that seemed useful, especially the transforms). Tested code - results was: it worked!

October 6, 2014 - C+A: Reviewed code so far, attempted to implement on computers.  Serious issues with launching it on their computers.

October 7, 2014 - All: In class we got everyone's workspace up and running.  Then we wrote pseudocode to implement at a meeting to be held later today.  At that next meeting we discussed the use of Bayes.  We decided to keep the current bayes set-up, and played with the spectrum.  We have a plan on logging data into a database once it is read, and as it fades the new color of the database log will come through.  After this, we will work on robot localization, by setting up a simple particle filter (maybe) and relate it to our generated map.

October 10, 2014 - All: In class Amanda worked on getting dynamic obstacles develped in Gazebo.  Claire and Victoria worked on adjusting the map display, and identifying the spaces in which Bayesian updates should be made and developed.  In the evening we played with the bayesian equations and attempted to make a quickly updating map.  This was achieved by updating the hits within the particle for loop, and limiting how 'unlikely' something was with being present.  By doing this, you can create the effect of a more dynamic environment, by assuming some arbitrarily small liklihood that SOMETHING will be in a location, no matter how many times you read it as blank.

October 11, 2014 - C+A: Added dynamic obstacles and second map that has only them plotted on it. Plots on orginal map as well. Obstacles dissapear after 15 cycles.

October 12, 2014 - All: We made the thingy work!  Mapping is still rough, can optimize tomorrow.  Adjusted things like top cap of odds, constants, adjusting pathing, etc.
"""
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
from numpy.random import random_sample, rand
from numpy import vstack, array
from scipy.cluster.vq import kmeans, vq
from pylab import plot, show
from matplotlib.pyplot import imshow
from sklearn.neighbors import NearestNeighbors

class TransformHelpers:
	""" Helper functions for making transformations of data from the RunMapping
	world to the script and back.  Will only be useful for us if we add an autonomy
	function to the robot """


	@staticmethod
	def convert_pose_to_xy_and_theta(pose):
		""" Convert pose (geometry_msgs.Pose) to a (x,y,yaw) tuple """
		orientation_tuple = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
		angles = euler_from_quaternion(orientation_tuple)
		return (pose.position.x, pose.position.y, angles[2])


""" Difficulty Level 2 """
class RunMapping:
	""" Stores an occupancy field for an input map.  An occupancy field returns the distance to the closest
		obstacle for any coordinate in the map
		Attributes:
			map: the map to localize against (nav_msgs/OccupancyGrid)
			closest_occ: the distance for each entry in the OccupancyGrid to the closest obstacle
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

		self.dyn_obs=[]
		self.rapid_appear = set()
		self.counter=0

		#Giving initial hypotheses to the system
		self.p_occ = 0.5*np.ones((self.n, self.n)) #50-50 chance of being occupied
		self.odds_ratio_hit = 3.0 #this is arbitrary, can re-assign
		self.odds_ratio_miss = 0.3 #this is arbitrary, can reassign
		#TODO: Evaluate these - what do we need to change in order to make this more friendly to our version?  Potential changes:
		#Whenever there is an adjustment to self.odds_ratio_miss, update an odds ratio that implies dynamicness


		#calculates odds based upon hit to miss, equal odds to all grid squares
		self.odds_ratios = (self.p_occ)/(1-self.p_occ)*np.ones((self.n, self.n))
		#calculate initial past odds_ratio
		self.past_odds_ratios = (self.p_occ)/(1-self.p_occ)*np.ones((self.n, self.n))
		#TODO: Make sure that this is still an accurate representation of how probabilities work in our system.  Make appropriate additions/adjustments for the dynamic obstacle piece
		
		#write laser pubs and subs
		rospy.Subscriber("scan", LaserScan, self.scan_received, queue_size=1)
		self.pub = rospy.Publisher("map", OccupancyGrid)

		#note - in case robot autonomy is added back in
		self.tf_listener = TransformListener()	
	
	def get_closest_obstacle_distance(self,x,y): #CHANGE TO get_closest_obstacle_path
		""" Compute the closest obstacle to the specified (x,y) coordinate in the map.  If the (x,y) coordinate
			is out of the map boundaries, nan will be returned. """
		pass
			# x_coord = int((x - self.map.info.origin.position.x)/self.map.info.resolution)
			# y_coord = int((y - self.map.info.origin.position.y)/self.map.info.resolution)
			# # check if we are in bounds
			# if x_coord > self.map.info.width or x_coord < 0:
			# 	return float('nan')
			# if y_coord > self.map.info.height or y_coord < 0:
			# 	return float('nan')
			# ind = x_coord + y_coord*self.map.info.width
			# if ind >= self.map.info.width*self.map.info.height or ind < 0:
			# 	return float('nan')
			# return self.closest_occ[ind]

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
						#self.p_occ[x_ind, y_ind] *= self.p_occ[x_ind, y_ind] * self.odds_ratio_miss/self.odds_ratio_hit
						marked.add((x_ind, y_ind))
						#print 'New Point'
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
					if self.odds_ratios[i,j] < 1/5.0:
						map.data[idx] = 0 #makes the gray
					elif self.odds_ratios[i,j] >= 1/5.0 < 0.5:
						map.data[idx] = 25
					elif self.odds_ratios[i,j] > 0.5:
						map.data[idx] = 100 #makes the black walls
					else:
						map.data[idx] = -1 #makes unknown
			self.pub.publish(map)
			#TODO: Change display such that we're not just looking at the ratio, but mapping the dynamic archive and current readings

		image = np.zeros((self.odds_ratios.shape[0], self.odds_ratios.shape[1],3))
		image2 = np.zeros((self.odds_ratios.shape[0], self.odds_ratios.shape[1],3))

		self.counter+=1

		x_odom_index = int((self.odom_pose[0] - self.origin[0])/self.resolution)
		y_odom_index = int((self.odom_pose[1] - self.origin[1])/self.resolution)

		self.pose.append((x_odom_index, y_odom_index))

		#.shape() comes from being related to the np class
		for i in range(image.shape[0]):
			for j in range(image.shape[1]):
				#print self.past_odds_ratios[i,j]
				#print self.odds_ratios[i,j]
				#the thing that just rapidly appeared, disappeared!
				delta = (self.odds_ratios[i,j]-self.past_odds_ratios[i,j])
				if (delta < 0.0) and (i,j) in self.rapid_appear:
					self.dyn_obs.append((i,j,self.counter))

				#whoa buddy, a thing just appeared
				if delta > 1000.0 and (i,j) not in self.rapid_appear:
					self.rapid_appear.add((i,j))

				if self.odds_ratios[i,j] < 1/50.0:
					image[i,j,:] = 1.0 #makes open space
				elif self.odds_ratios[i,j] >= 1/50.0 and self.odds_ratios[i,j] <4/5.0:
					image[i,j,:] = (0, 255, 0)
				elif self.odds_ratios[i,j] > 50.0:
					image[i,j,:] = (0, 0, 255) #makes walls
				else:
					image[i,j,:] = 0.5 #not read
					

		if len(self.dyn_obs)>0:
			for point in self.dyn_obs:
				if (self.counter-point[2])<=100:
					image2[point[0],point[1]] = (255,0,255) #makes old/dynamic shapes on other map

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