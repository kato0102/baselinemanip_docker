import sys
sys.path.append('/opt/python')

import relay_node
# from relay_node import relayToROS
# from relay_node import relayFromROS
from relay_numpy_node import relayToROS
from relay_numpy_node import relayFromROS
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import Image, JointState  
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from control_msgs.msg import JointTrajectoryControllerState 
from cv_bridge import CvBridge
import rospy
import numpy as np
# ★ RobotInterface の読み込み（あなたの環境と同じやり方）
exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())
ri = RobotInterface('/choreonoid_ws/src/irsl_hsr_pkgs/irsl_hsr_model/hsrb/robot_interface.yaml',
                    connection=True)
def twist_array_to_msg(ary):
    msg = Twist()
    msg.linear.x =  ary[0]
    msg.linear.y =  ary[1]
    msg.angular.z = ary[2]
    return msg
def array_to_jointtraj(ary, joint_names, duration=1.0):
    """numpy → JointTrajectory"""
    traj = JointTrajectory()
    traj.joint_names = joint_names

    point = JointTrajectoryPoint()
    point.positions = ary.tolist()
    point.time_from_start = rospy.Duration(duration)

    traj.points = [point]
    return traj
def odom_msg_to_array(msg):
    ary =np.array((msg.twist.twist.linear.x,
                   msg.twist.twist.linear.y,
                   msg.twist.twist.angular.z), dtype='float32')
    return ary

def image_msg_to_array(msg):
    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(msg)
    return cv_img

# ★ hand_camera 用（内容は head_camera と同じく OpenCV BGR画像に変換）
def hand_image_msg_to_array(msg):
    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(msg)
    return cv_img



STATE_JOINT_ORDER = [
    "arm_flex_joint",
    "arm_lift_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "hand_motor_joint",
]

def joint_states_msg_to_array(msg):
    # name → index
    name_to_idx = {name: i for i, name in enumerate(msg.name)}

    vals = []
    for n in STATE_JOINT_ORDER:
        if n in name_to_idx:
            vals.append(msg.position[name_to_idx[n]])
        else:
            # 念のため無いときは0で埋める
            vals.append(0.0)

    return np.asarray(vals, dtype="float32")


# ★ JointTrajectoryControllerState → [actual, desired, error] を結合
def jtc_state_msg_to_array(msg):
    actual = np.asarray(msg.actual.positions, dtype="float32")
    desired = np.asarray(msg.desired.positions, dtype="float32")
    error = np.asarray(msg.error.positions, dtype="float32")
    return np.concatenate([actual, desired, error], axis=0)
def jointtraj_msg_to_array(msg):
    """JointTrajectory → numpy に変換（positions のみ使用）
       今回は points[0] のみ利用（HSRは1ポイント指令がほとんど）
    """
    if len(msg.points) == 0:
        return None
    p = msg.points[0]
    return np.asarray(p.positions, dtype='float32')

r_twist = relayToROS('ice_twist', '/hsrb/command_velocity', Twist, twist_array_to_msg)
r_odom  = relayFromROS('ice_odom', '/hsrb/odom', Odometry, odom_msg_to_array)
r_image = relayFromROS('ice_image', '/hsrb/head_rgbd_sensor/rgb/image_rect_color', Image, image_msg_to_array)
r_hand_image = relayFromROS(
    'ice_hand_image',
    '/hsrb/hand_camera/image_raw',
    Image,
    hand_image_msg_to_array,
)

# joint_states
r_joint_states = relayFromROS(
    'ice_joint_states',
    '/hsrb/joint_states',
    JointState,
    joint_states_msg_to_array,
)

# head_trajectory_controller/state
r_head_state = relayFromROS(
    'ice_head_state',
    '/hsrb/head_trajectory_controller/state',
    JointTrajectoryControllerState,
    jtc_state_msg_to_array,
)

# arm_trajectory_controller/state
r_arm_state = relayFromROS(
    'ice_arm_state',
    '/hsrb/arm_trajectory_controller/state',
    JointTrajectoryControllerState,
    jtc_state_msg_to_array,
)
r_arm_cmd = relayFromROS(
    'ice_arm_cmd',
    '/hsrb/arm_trajectory_controller/command',
    JointTrajectory,
    jointtraj_msg_to_array
)
ARM_JOINT_NAMES = ['arm_flex_joint','arm_lift_joint',  'arm_roll_joint',
                   'wrist_flex_joint', 'wrist_roll_joint']

r_arm_cmd_out = relayToROS(
    'ice_arm_cmd_out',
    '/hsrb/arm_trajectory_controller/command',
    JointTrajectory,
    lambda ary: array_to_jointtraj(ary, ARM_JOINT_NAMES)
)
r_head_cmd = relayFromROS(
    'ice_head_cmd',
    '/hsrb/head_trajectory_controller/command',
    JointTrajectory,
    jointtraj_msg_to_array
)

HEAD_JOINT_NAMES = ['head_pan_joint', 'head_tilt_joint']

r_head_cmd_out = relayToROS(
    'ice_head_cmd_out',
    '/hsrb/head_trajectory_controller/command',
    JointTrajectory,
    lambda ary: array_to_jointtraj(ary, HEAD_JOINT_NAMES)
)
r_gripper_cmd = relayFromROS(
    'ice_gripper_cmd',
    '/hsrb/gripper_controller/command',
    JointTrajectory,
    jointtraj_msg_to_array
)

GRIPPER_JOINT_NAMES = ['hand_motor_joint']

r_gripper_cmd_out = relayToROS(
    'ice_gripper_cmd_out',
    '/hsrb/gripper_controller/command',
    JointTrajectory,
    lambda ary: array_to_jointtraj(ary, GRIPPER_JOINT_NAMES)
)

r_twist.main('irsl_relay_twist')
r_odom .main('irsl_relay_odom')
r_image.main('irsl_relay_image')
r_hand_image.main('irsl_relay_hand_image')       # ★ 追加
r_joint_states.main('irsl_relay_joint_states')    # ★ 追加
r_head_state.main('irsl_relay_head_state')       # ★ 追加
r_arm_state.main('irsl_relay_arm_state')  
r_arm_cmd.main('relay_arm_cmd')
r_arm_cmd_out.main('relay_arm_cmd_out')

r_head_cmd.main('relay_head_cmd')
r_head_cmd_out.main('relay_head_cmd_out')

r_gripper_cmd.main('relay_gripper_cmd')
r_gripper_cmd_out.main('relay_gripper_cmd_out')
rospy.spin()
