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
    name_arm_state,
    name_head_state,
    name_odom,
    # name_wrench,
    name_joint_state,
    name_cam_head_rgb,
    name_cam_hand,
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

state_names = [
    "arm_flex_joint",
    "arm_lift_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    ##
    "head_pan_joint",
    "head_tilt_joint",
    "hand_motor_joint",
    ##
    "base_roll_joint",
    ]

action_names = [
    "arm_flex_joint",
    "arm_lift_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "head_pan_joint",
    "head_tilt_joint",
    ]
##
def makeState(msg_joint_state):
    data = {}
    for n, p in zip(msg_joint_state.name,  msg_joint_state.position):
        data[n] = p
    res = []
    for n in state_names:
        res.append(data[n])
    return np.array(res)

def makeAction(msg_arm_traj, msg_head_traj, msg_joint_state):
    data = {}
    for n, p in zip(msg_arm_traj.joint_names, msg_arm_traj.desired.positions):
        data[n] = p
    for n, p in zip(msg_head_traj.joint_names, msg_head_traj.desired.positions):
        data[n] = p
    res = []
    for n in action_names:
        res.append(data[n])
    ## hot fix for gripper
    idx = msg_joint_state.name.index("hand_motor_joint")
    res.append(msg_joint_state.position[idx])
    return np.array(res)

def mainFunction(bag_file, pkl_name, rate=10.0): ## add skip or rate
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

    ## store all messages
    all_msgs = {}
    for tname in use_topics:
        if not tname in topics:
            continue
        useHeader = False
        if hasattr(topic_class[ topics[tname].msg_type ]['class'], 'header'):
            useHeader = True
        ##
        lst = []
        # func = topic_class[ topics[tname].msg_type ]['func'] if 'func' in  topic_class[ topics[tname].msg_type ] else None
        # print(tname, func)
        for _ , msg, t in bag.read_messages(topics=[tname]):
            print('.', end='', flush=True)
            if useHeader:
                tm = rospy.Time(secs=msg.header.stamp.secs, nsecs=msg.header.stamp.nsecs).to_sec()
            else:
                tm = t.to_sec()
            #if func is not None:
            #    msg = func(msg)
            lst.append( (tm, msg,) )
        print('!')
        all_msgs[tname] = lst

    ## find nearest messages based on time
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

    ### remove None
    doing = True
    while doing:
        remove_idx = -1
        for key, lst in final_msgs.items():
            for idx, l in enumerate(lst):
                if l[0] is None:
                    remove_idx = idx
                    break
            if remove_idx >=0:
                break
        if remove_idx >=0:
            ## remove remove_idx
            print('remove : ', idx)
            for key, lst in final_msgs.items():
                del lst[remove_idx]
        else:
            ## all data is not None
            doing = False

    ### TODO: skip or rate
    if rate is not None:
        dur = 1 / rate
        indices = [0]
        tm = final_msgs['T']
        prev = tm[0][0]
        for idx in range(len(tm)):
            cur = tm[idx][0]
            if cur - prev >= dur:
                indices.append(idx)
                prev = cur
        tmp = final_msgs
        final_msgs = {}
        for k in tmp.keys():
            res = []
            vals = tmp[k]
            for idx in indices:
                res.append(vals[idx])
            final_msgs[k] = res

    ### convert msgs -> np.array
    arrays = {}
    sz = len(final_msgs['T'])
    print(sz)
    arrays['state_pos']   = []
    arrays['action_pos']  = []
    arrays['state_odom']  = []
    arrays['action_odom'] = []
    arrays['head_image']  = []
    arrays['hand_image']  = []
    arrays['T'] = []
    for idx in range(sz):
        state = makeState(
            final_msgs[name_joint_state][idx][1],
        )
        action = makeAction(
            final_msgs[name_arm_state][idx][1],
            final_msgs[name_head_state][idx][1],
            final_msgs[name_joint_state][idx][1],
        )
        state_odom = odom_to_data( final_msgs[name_odom][idx][1] )
        action_odom = twist_to_data( final_msgs[name_command_twist][idx][1] )
        head_image = _from_rosImage( final_msgs[name_cam_head_rgb][idx][1] )
        hand_image = _from_rosImage( final_msgs[name_cam_hand][idx][1] )
        arrays['state_pos' ].append(state)
        arrays['action_pos'].append(action)
        arrays['state_odom' ].append(state_odom)
        arrays['action_odom'].append(action_odom)
        arrays['head_image'].append(head_image)
        arrays['hand_image'].append(hand_image)
        arrays['T'].append(final_msgs['T'][idx][0])

    with open(pkl_name, 'wb') as f:
        pickle.dump(arrays, f)

    return arrays

## ipython
## %autoindent
## exec(open('python/convert_bag_to_pickle.py').read())
## _ = mainFunction('2025-12-03-16-39-15.bag', 'hsrbag000.pkl')
##
