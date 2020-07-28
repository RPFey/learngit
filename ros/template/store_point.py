#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import ros_numpy
import numpy as np
import os
import cv2

import sys, select, termios, tty

bridge = CvBridge()

img_base = '/home/cag/ubot_sim/images'
point_base = '/home/cag/ubot_sim/points'
pose_base = '/home/cag/ubot_sim/poses'

def store(point, pose, image):
    time_stamp = str(image.header.stamp.nsecs)
    dot = time_stamp.find('.')
    time_stamp = time_stamp[:dot]

    img_name = os.path.join(img_base, time_stamp+'.jpg')
    cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
    cv2.imwrite(img_name, cv_image)

    points = ros_numpy.numpify(point)
    x_coor = points['x'].reshape(-1)
    y_coor = points['y'].reshape(-1)
    z_coor = points['z'].reshape(-1)

    pos_vec = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.rotation.x,
               pose.pose.position.y, pose.pose.position.z, pose.pose.position.w]

    pos_vec = np.array(pos_vec)

    point_name = os.path.join(point_base, time_stamp+'.npz')
    np.savez(point_name, x=x_coor, y=y_coor, z=z_coor, pos=pos_vec)
   
    print "messages saved"


if __name__=="__main__":
    rospy.init_node('robot_teleop')
    point_cloud = message_filters.Subscriber('/camera/depth/points', PointCloud2)
    relative_pose = message_filters.Subscriber('/relative_pose', PoseStamped)
    rgb_img = message_filters.Subscriber('/camera/rgb/image_raw', Image)
    sync = message_filters.TimeSynchronizer([point_cloud, relative_pose, rgb_img], 10)
    sync.registerCallback(store)
    print "node initiate"
    rospy.spin()

