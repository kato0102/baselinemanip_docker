# %autoindent
#
# under ROS environment
#
import rosbag
import rospy
import pickle

name_arm_state  = '/hsrb/arm_trajectory_controller/state'
name_head_state = '/hsrb/head_trajectory_controller/state'
name_odom       = '/hsrb/odom'
name_wrench     = '/hsrb/wrist_wrench/raw'
name_joint_state = '/hsrb/joint_states'
name_command_twist = '/hsrb/command_velocity'

name_cam_head_rgb   = '/hsrb/head_rgbd_sensor/rgb/image_rect_color' ## main-camera
hame_cam_head_depth = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
name_cam_hand       = '/hsrb/hand_camera/image_raw'

#
# determine time steps with this message
#
main_topic = name_command_twist

use_topics = (
    name_command_twist,
    # name_arm_state,
    # name_head_state,
    name_odom,
    # name_wrench,
    # name_joint_state,
    name_cam_head_rgb,
)

def twist_to_data(twist):
    ## numpy version
    return np.array((twist.linear.x, twist.linear.y, twist.angular.z), dtype='float32')
    ## list version
    #return (twist.linear.x, twist.linear.y, twist.angular.z,)

def odom_to_data(odom):
    return twist_to_data(odom.twist.twist)

### camera
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np

def _from_rosImage(msg):
    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(msg)
    ##
    #return cv_img.tolist()
    ## numpy version
    return cv_img

def find_nearest(lst, tm, start=0):
    size = len(lst)
    prev_t, prev_msg = lst[0]
    if tm <= prev_t:
        return -1, prev_t, prev_msg
    for i in range(start, size):
        t, msg = lst[i]
        if tm > prev_t and tm <= t:
            if tm - prev_t > t - tm:
                return i, t, msg
            else:
                return i - 1, prev_t, prev_msg
    return None, None, None ## last-one

convertFunctions = {
    'nav_msgs/Odometry':    odom_to_data,
    'geometry_msgs/Twist':  twist_to_data,
    'sensor_msgs/Image':    _from_rosImage,
    }

def mainFunction(bag_file, pkl_name):
    ## open bag
    bag = rosbag.Bag(bag_file)
    topic_types, topics = bag.get_type_and_topic_info()

    ## parse classes
    topic_class = {}
    for i, mname in enumerate(topic_types):
        mg = mname.split('/')
        evstr = f'from {mg[0]}.msg import {mg[1]} as msg{i:0>3}'
        print(evstr)
        exec(evstr)
        tmp = { 'class': eval(f'msg{i:0>3}') }
        if mname in convertFunctions:
            tmp['func'] = convertFunctions[mname]
        topic_class[mname] = tmp

    ## main - time
    main_msgs = []
    print(main_topic)
    useHeader = False
    if hasattr(topic_class[ topics[main_topic].msg_type ]['class'], 'header'):
        useHeader = True
    for _ , msg, t in bag.read_messages(topics=[main_topic]):
        print('.', end='', flush=True)
        if useHeader:
            tm = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs).to_sec()
        else:
            tm = t.to_sec()
        main_msgs.append( (tm, tm,) )
    print('!')

    ##
    all_msgs = {}
    for tname in use_topics:
        if not tname in topics:
            continue
        useHeader = False
        if hasattr(topic_class[ topics[tname].msg_type ]['class'], 'header'):
            useHeader = True
        ##
        lst = []
        func = topic_class[ topics[tname].msg_type ]['func'] if 'func' in  topic_class[ topics[tname].msg_type ] else None
        print(tname, func)
        for _ , msg, t in bag.read_messages(topics=[tname]):
            print('.', end='', flush=True)
            if useHeader:
                tm = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs).to_sec()
            else:
                tm = t.to_sec()
            if func is not None:
                msg = func(msg)
            lst.append( (tm, msg,) )
        print('!')
        all_msgs[tname] = lst

    ##
    final_msgs = {}
    for tname in use_topics:
        if not tname in all_msgs:
            continue
        lst = all_msgs[tname]
        idx = 0
        res = []
        print(tname)
        for tm, _ in main_msgs:
            idx, t, msg = find_nearest(lst, tm, start = idx)
            res.append( (t, msg, idx, ) )
            if idx is None:
                idx = len(lst)-1
            if idx < 0:
                idx = 0
        final_msgs[tname] = res
    final_msgs['T'] = main_msgs
    #final_msgs['__topic_types'] = topic_types
    #final_msgs['__topics'] = topics

    with open(pkl_name, 'wb') as f:
        pickle.dump(final_msgs, f)

    return final_msgs

##
## mainFunction('test.bag', 'hsrbag000.pkl')
##
