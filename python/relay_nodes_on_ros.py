import sys
sys.path.append('/opt/python')

import relay_node
# from relay_node import relayToROS
# from relay_node import relayFromROS
from relay_numpy_node import relayToROS
from relay_numpy_node import relayFromROS

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import numpy as np

def twist_array_to_msg(ary):
    msg = Twist()
    msg.linear.x =  ary[0]
    msg.linear.y =  ary[1]
    msg.angular.z = ary[2]
    return msg

def odom_msg_to_array(msg):
    ary =np.array((msg.twist.twist.linear.x,
                   msg.twist.twist.linear.y,
                   msg.twist.twist.angular.z), dtype='float32')
    return ary

def image_msg_to_array(msg):
    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(msg)
    return cv_img

r_twist = relayToROS('ice_twist', '/hsrb/command_velocity', Twist, twist_array_to_msg)
r_odom  = relayFromROS('ice_odom', '/hsrb/odom', Odometry, odom_msg_to_array)
r_image = relayFromROS('ice_image', '/hsrb/head_rgbd_sensor/rgb/image_rect_color', Image, image_msg_to_array)

r_twist.main('irsl_relay_twist')
r_odom .main('irsl_relay_odom')
r_image.main('irsl_relay_image')

rospy.spin()
