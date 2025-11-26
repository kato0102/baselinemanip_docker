import sys
sys.path.append('/opt/python')

#
# rollout side
#

from irsl_iceoryx2 import recvNumpy
from irsl_iceoryx2 import sendNumpy
# from nav_msgs.msg import Odometry
import numpy as np

### subscriber
sub_odom = recvNumpy('ice_odom')
sub_img  = recvNumpy('ice_image')

### publisher
pub_twist = sendNumpy('ice_twist')

# subscribe state
odom = sub_odom.getLastAry()
img  = sub_img.getLastAry()

# publish action
np_action = np.array([0.1, 0, 0])
pub.sendAry(np_action)
