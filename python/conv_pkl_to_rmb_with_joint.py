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
if __name__ == "__main__":
    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser(
        description="Convert HSR pickle data (.pkl) to RMB format (.rmb). "
                    "If pkl_path is a directory, all *.pkl under it will be converted."
    )
    parser.add_argument(
        "pkl_path",
        type=str,
        help="input pickle file OR directory containing pickle files"
    )
    parser.add_argument(
        "rmb_name",
        type=str,
        nargs="?",
        help="output rmb file (used only when pkl_path is a single pickle file)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to store output RMB files (directory mode only)"
    )

    args = parser.parse_args()

    # ==========================================================
    #  DIRECTORY MODE
    # ==========================================================
    if os.path.isdir(args.pkl_path):
        input_dir = args.pkl_path
        pkl_files = sorted(
            glob.glob(os.path.join(input_dir, "**", "*.pkl"), recursive=True)
        )

        if not pkl_files:
            print("[WARN] No .pkl files found.")
            raise SystemExit(0)

        if args.outdir is None:
            print("[ERROR] Directory mode requires --outdir OUTPUT_FOLDER")
            print("Example:")
            print("  python convert_pickle_to_rmb.py pkls/ --outdir rmbs/")
            raise SystemExit(1)

        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)

        print(f"[INFO] Directory mode")
        print(f"[INFO] Input pkls : {input_dir}")
        print(f"[INFO] Output dir : {outdir}")
        print(f"[INFO] Found {len(pkl_files)} pkl files")

        for pkl_file in pkl_files:
            base = os.path.basename(pkl_file)          # xxx.pkl
            rmb_filename = os.path.splitext(base)[0] + ".rmb"
            out_path = os.path.join(outdir, rmb_filename)

            print(f"[INFO] Converting: {pkl_file} -> {out_path}")
            try:
                convert_pickle_to_rmb(
                    pkl_file,
                    out_path,
                    data_names, 
                )
            except Exception as e:
                print(f"[ERROR] Failed to convert {pkl_file}: {e}")

    # ==========================================================
    #  SINGLE-FILE MODE
    # ==========================================================
    else:
        if args.rmb_name is None:
            print("[ERROR] Single-file mode requires rmb_name.")
            print("Usage:")
            print("  python convert_pickle_to_rmb.py input.pkl output.rmb")
            raise SystemExit(1)

        print("[INFO] File mode")
        print(f"[INFO] input  pkl : {args.pkl_path}")
        print(f"[INFO] output rmb : {args.rmb_name}")

        convert_pickle_to_rmb(
            args.pkl_path,
            args.rmb_name,
            data_names,

        )