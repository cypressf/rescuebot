#!/usr/bin/env python

import rospy, math
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

# Whee globals!
pub = None
aligned_to_wall = 'Not aligned'

# Sets default target distance to wall (in meters)
target_dist = 1.0	
dist_to_wall = -1.0

def approach_wall(msg):
	"""Approach wall, ending at specified target distance"""
	global target_dist, dist_to_wall
	global front_scan
	
	pub = [msg.ranges[x] for x in range(5) if range_is_valid(msg.ranges[x])]

	print dist_to_wall
	if len(temp) > 0:
		dist_to_wall = sum(front_scan)/len(front_scan)
		pub.publish(Twist(linear=Vector3(x=0.3*(dist_to_wall-target_dist))))
	else:
		dist_to_wall = -1.0


def wall_nearby(msg):
	"""Determine if there is a wall within nearby range. If so, determine nearest side (LEFT or RIGHT) and orient towards."""
	global aligned_to_wall
	global pub 

	# Sketchy stuff to isolate bad values without destroying indices -- to be done properly later  
	tol = 0.1 # Because scan data is not entirely accurate or consistent

	# Finds shortest laserscan range detected, and assumes it leads to wall
	full_data = [msg.ranges[x] if range_is_valid(msg.ranges[x]) else 1e3 for x in range(360)]
	min_dist = min(full_data)
	theta = full_data.index(min_dist)

	# Constructs a second line 5 degrees off, checks to see if it forms proper right triangle
	dist_hyp = msg.ranges[(theta+5)%360]
	dist_opp = math.tan(math.radians(5)/min_dist)
	if abs(math.sqrt(min_dist**2 + dist_opp**2) - dist_hyp) <= tol:
		print 'Close enough -- wall detected'
	else:
		print 'False alarm'

	print theta

	# Rotates robot, so that nearest side (90=left, 270=right) is aligned with wall (represented by theta)
	if theta in range(85,95):
		print 'Basically aligned to left'
		aligned_to_wall = 'LEFT' 
	elif theta in range(265,275):
		print 'Basically aligned to right'
		aligned_to_wall = 'RIGHT'
	else:
		if (theta>180):
			print 'Turning right'
			pub.publish(Twist(angular=Vector3(z=-0.2)))
		else:
			print 'Turning left'
			pub.publish(Twist(angular=Vector3(z=0.2)))

	print '***************************************************'

def follow(wall_dir, msg):
	"""Implement wall-following behavior (parallel to wall) for specified wall direction"""
	global pub, aligned_to_wall

	base_turn_speed = 1
	tol = 1.5e-2

	# The scanning range that corresponds to each side (assuming +/-45 degrees)
	scan_ranges = {
		'LEFT': (range(45, 60), range(120, 135)),
		'RIGHT': (range(225, 230), range(310, 315)),
	}

	# Looks at scans from side closest to specified wall direction and calculates difference in scans +/- 45 degrees from presumed center
	side_front = [msg.ranges[x] for x in scan_ranges[wall_dir][0] if range_is_valid(msg.ranges[x])]
	side_back = [msg.ranges[x] for x in scan_ranges[wall_dir][1] if range_is_valid(msg.ranges[x])]	
	print('********* SIDE FRONT', side_front, '********')
	print('********* SIDE BACK', side_back, '********')

	if len(side_front) == 0 or len(side_back) == 0:
		aligned_to_wall = 'Not aligned'
		return
	
	side_dif = (sum(side_front)/len(side_front)) - (sum(side_back)/len(side_back))

	# If difference in sides is below tolerance, go forward.  Elsewise, turn in logical direction
	if abs(side_dif) <= tol:
		print ('Close enough -- theoretically moving forward')
		pub.publish(Twist(linear=Vector3(x=.2)))
	else:
		if side_dif < 0:
			print('Theoretically turning right')
		else:
			print('Theoretically turning left')
		pub.publish(Twist(angular=Vector3(z=side_dif*base_turn_speed)))

	print 'Side difference: ', side_dif
	print '*********************************************************'

def range_is_valid(scan_msg):
	"""Returns True if laser scan message range is within valid range"""
	if scan_msg >= 0.1 and scan_msg < 7.0:
		return True

	return False

def scan_recieved(msg):
	"""Callback function for msg of type sensor_msgs/LaserScan"""
	global pub 
	global aligned_to_wall

	base_turn_speed = 4
	tol = 1.5e-2

	if len(msg.ranges) != 360:
		print 'Unexpected laser scan message'
		return

	# follow(wall_nearby(), msg)
	# follow('LEFT', msg)
	# approach_wall(msg)

	if aligned_to_wall == 'Not aligned':
		wall_nearby(msg)
	else:
		follow(aligned_to_wall, msg)


def test():
	"""Main run loop for testing."""
	global pub
	# Sets to publish to 'cmd_vel' topic and subscribe to laserscan rangefinder ('scan') topic
	pub = rospy.Publisher('/cmd_vel', Twist)
	sub = rospy.Subscriber('/scan', LaserScan, scan_recieved)
	# Initiates as 'wall_follow' node
	rospy.init_node('wall_follow', anonymous=True)
	# Sets publish rate to 10 hz 
	r = rospy.Rate(10)

	while not rospy.is_shutdown():
		r.sleep()


if __name__ == '__main__':
	try:
		test()

	except rospy.ROSInterruptException: pass