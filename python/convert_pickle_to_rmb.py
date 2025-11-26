##
## under venv (without ROS)
##

from robo_manip_baselines.common import RmbData, DataKey
from robo_manip_baselines.common.manager.DataManager import DataManager
# import rospy
import pickle

data_names = {
    DataKey.get_rgb_image_key('head') : '/hsrb/head_rgbd_sensor/rgb/image_rect_color',
    DataKey.MEASURED_MOBILE_OMNI_VEL : '/hsrb/odom',
    DataKey.COMMAND_MOBILE_OMNI_VEL : '/hsrb/command_velocity',
    DataKey.TIME : 'T',
    }

def make_rmb_data(rmb_name, pkl_data, data_names, demo_name='demo0', task_name='task0'):
    data_manager=DataManager(None, demo_name, task_name)
    #
    length = len(pkl_data['T'])
    for i in range(length):
        print('.', end='', flush=True)
        for rmb_k, data_k in data_names.items():
            data_manager.append_single_data(rmb_k, pkl_data[data_k][i][1])
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
