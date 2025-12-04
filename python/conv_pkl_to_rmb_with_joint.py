##
## under venv (without ROS)
##

from robo_manip_baselines.common import RmbData, DataKey
from robo_manip_baselines.common.manager.DataManager import DataManager
# import rospy
import pickle

#> 'COMMAND_DATA_KEYS',
#> 'COMMAND_EEF_POSE',
#> 'COMMAND_EEF_POSE_REL',
#> 'COMMAND_EEF_VEL',
#> 'COMMAND_EEF_WRENCH',
#> 'COMMAND_GRIPPER_JOINT_POS',
#> 'COMMAND_GRIPPER_JOINT_POS_REL',
#> 'COMMAND_JOINT_POS',
#> 'COMMAND_JOINT_POS_REL',
#> 'COMMAND_JOINT_TORQUE',
#> 'COMMAND_JOINT_VEL',
#> 'COMMAND_MOBILE_OMNI_VEL',
#> 'MEASURED_DATA_KEYS',
#> 'MEASURED_EEF_POSE',
#> 'MEASURED_EEF_POSE_REL',
#> 'MEASURED_EEF_VEL',
#> 'MEASURED_EEF_WRENCH',
#> 'MEASURED_GRIPPER_JOINT_POS',
#> 'MEASURED_GRIPPER_JOINT_POS_REL',
#> 'MEASURED_JOINT_POS',
#> 'MEASURED_JOINT_POS_REL',
#> 'MEASURED_JOINT_TORQUE',
#> 'MEASURED_JOINT_VEL',
#> 'MEASURED_MOBILE_OMNI_VEL',
#> 'REWARD',
#> 'TIME',

data_names = {
    ## camera
    DataKey.get_rgb_image_key('head') : 'head_image',
    DataKey.get_rgb_image_key('hand') : 'hand_image',
    ## joint_pos
    DataKey.MEASURED_JOINT_POS : 'state_pos',
    DataKey.COMMAND_JOINT_POS : 'action_pos',
    ## omni
    DataKey.MEASURED_MOBILE_OMNI_VEL : 'state_odom',
    DataKey.COMMAND_MOBILE_OMNI_VEL : 'action_odom',
    ## time
    DataKey.TIME : 'T',
    }

data_functions = {

}

def make_rmb_data(rmb_name, pkl_data, data_names, demo_name='demo0', task_name='task0'):
    data_manager=DataManager(None, demo_name, task_name)
    #
    length = len(pkl_data['T'])
    for i in range(length):
        print('.', end='', flush=True)
        for rmb_k, data_k in data_names.items():
            print(rmb_k)
            data_manager.append_single_data(rmb_k, pkl_data[data_k][i])
    print('!')
    #destination
    data_manager.save_data(rmb_name)

def convert_pickle_to_rmb(pkl_name, rmb_name, data_names):
    with open(pkl_name, 'rb') as f:
        pkl_data = pickle.load(f)
    make_rmb_data(rmb_name, pkl_data, data_names)

##
## convert_pickle_to_rmb('hsrbag000.pkl', 'hsr000.rmb', data_names)
##
