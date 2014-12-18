#!/usr/bin/env python

import rospy
import dynamic_reconfigure.server
from rescuebot.cfg import ControllerConfig


def main():
    rospy.init_node("dynamic_reconfigure_server")
    dynamic_reconfigure.server.Server(ControllerConfig, reconfigure)
    while not rospy.is_shutdown():
        rospy.spin()


def reconfigure(config, level):
    return config  # Returns the updated configuration.

if __name__ == '__main__':
    main()