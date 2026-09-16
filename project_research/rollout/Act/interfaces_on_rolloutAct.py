# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("/opt/python")   # irsl_iceoryx2 用

from irsl_manip_libs.irsl_iceoryx2 import recvNumpy, sendNumpy
import numpy as np
import time
import torch
from torchvision.transforms import v2
from robo_manip_baselines.common.utils.DataUtils import normalize_data
from robo_manip_baselines.common import denormalize_data
import cv2
import threading
import argparse
# ----- ここで RolloutAct 用の引数をねじ込む（checkpoint 指定） -----
# sys.argv = [
#     "interfaces_on_rollout.py",
#     "--checkpoint",
#     "/RoboManipBaselines/robo_manip_baselines/checkpoint/Act/"
#     "HsrPickrmb_Act_20251213_161351/policy_best.ckpt",
#     #"--skip", "6"
# ]
from rollout_Act import InteractiveRollout
def parse_args():
    parser = argparse.ArgumentParser(
        description="ACT rollout over iceoryx (HSR)"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to ACT policy checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        help="(optional) frame skip for ACT (forwarded to RolloutAct --skip)",
    )
    return parser.parse_args()


def main():
        # ===== 0) このスクリプト自身の引数をパース =====
    args = parse_args()

    # ----- ここで RolloutAct 用の引数をねじ込む（checkpoint / skip 指定） -----
    rollout_argv = [
        "interfaces_on_rollout.py",
        "--checkpoint",
        args.checkpoint,
    ]
    if args.skip is not None:
        rollout_argv += ["--skip", str(args.skip)]

    # RolloutAct が使う sys.argv を上書き
    sys.argv = rollout_argv
    # ========== 1) ACT policy のロード ==========
    rollout = InteractiveRollout()

    ### debug info TODO: => InteractiveRollout
    state_meta  = rollout.model_meta_info["state"]
    action_meta = rollout.model_meta_info["action"]
    state_dim   = rollout.state_dim
    action_dim  = rollout.action_dim
    print("[info] state_dim:", state_dim, "action_dim:", action_dim)
    ###
    state_keys  = rollout.model_meta_info.get("state_keys", None)
    action_keys = rollout.model_meta_info.get("action_keys", None)
    print("[info] state_keys:", state_keys)
    print("[info] action_keys:", action_keys)

    # ========== 2) iceoryx インタフェース ==========
    ### subscribe
    sub_hand_img    = recvNumpy("ice_hand_image")    # hand camera
    sub_joint_state = recvNumpy("ice_joint_states")  # joint positions
    ### publish
    pub_arm_cmd     = sendNumpy("ice_arm_cmd_out")
    pub_gripper_cmd = sendNumpy("ice_gripper_cmd_out")
    print("[info] start ACT rollout over iceoryx")

    # ========== 3) 次元の定義 ==========
    JOINT_STATE_DIM = 8
    # JOINT_STATE_DIM = 14 ## with torque
    JOINT_CMD_DIM   = 7  # command_joint_pos
    GRIPPER_CMD_DIM = 1
    ACTION_DIM      = JOINT_CMD_DIM + GRIPPER_CMD_DIM  # 7

    # ===== 共有変数（推論スレッドが書き込み、PUB スレッドが読み取り） =====
    cmd_lock = threading.Lock()
    latest_arm_cmd     = np.zeros((JOINT_CMD_DIM,),   dtype=np.float64)
    latest_gripper_cmd = np.zeros((GRIPPER_CMD_DIM,), dtype=np.float64)
    has_action         = False  # まだ一度も推論していない間は publish しない

    # ===== Publish ループ（PUB_INTERVAL ごとに呼ばれる “コールバック”的役割）=====
    PUB_INTERVAL = 1/10  # ここで pub 周期だけを制御（10Hz）

    def publish_loop():
        nonlocal latest_arm_cmd, latest_gripper_cmd, has_action
        try:
            while True:
                time.sleep(PUB_INTERVAL)
                with cmd_lock:
                    if not has_action:
                        continue
                    arm_cmd   = latest_arm_cmd.copy()
                    grip_cmd  = latest_gripper_cmd.copy()
                    has_action = False
                pub_arm_cmd.sendAry(arm_cmd)
                pub_gripper_cmd.sendAry(grip_cmd)
                # デバッグしたければここに print を置く
                print("[pub] arm:", arm_cmd, "grip:", grip_cmd)

        except KeyboardInterrupt:
            print("[publish_loop] interrupted")

    # ===== 推論ループ（可能な限り回し続ける） =====
    def inference_loop():
        nonlocal latest_arm_cmd, latest_gripper_cmd, has_action
        hand_img = None
        joint_state = None
        while True:
            # --- 1) 最新のセンサ値を取得 ---
            hand_img_latest =  sub_hand_img.getLastAry()
            joint_state_latest = sub_joint_state.getLastAry()
            hand_img = hand_img if hand_img_latest is None else hand_img_latest
            joint_state = joint_state if joint_state_latest is None else joint_state_latest
            # odom / 画像 / joint_state のどれかが未到着なら待つ
            if joint_state is None or hand_img is None:
                time.sleep(0.005)
                continue

            ###
            js = np.asarray(joint_state, dtype=np.float32).reshape(-1)
            if js.shape[0] > JOINT_STATE_DIM:
                js = js[:JOINT_STATE_DIM]
            elif js.shape[0] < JOINT_STATE_DIM:
                js = np.pad(js, (0, JOINT_STATE_DIM - js.shape[0]))

            state = np.concatenate([js], axis=0)

            # ===== 画像（head + hand の2カメラ） =====
            np_hand = np.asarray(hand_img, dtype=np.uint8)
            np_hand = cv2.cvtColor(np_hand, cv2.COLOR_BGR2RGB)

            images = [ np_hand ]

            # ===== ACT policy から action を計算（推論）=====
            action = rollout.step(state, images, do_plot=False)

            arm_cmd = action[:JOINT_CMD_DIM]
            gripper_cmd = action[JOINT_CMD_DIM:]

            # ===== 最新コマンドとして共有変数に書き込む（Publish スレッドが読む）=====
            with cmd_lock:
                latest_arm_cmd     = np.asarray(arm_cmd, dtype=np.float64)
                latest_gripper_cmd = np.asarray(gripper_cmd, dtype=np.float64)
                has_action         = True

            # デバッグ表示（推論側）
            print(
                "state(joint+odom):\n", state,
                " \n| arm:", latest_arm_cmd,
                "\n | grip:", latest_gripper_cmd,
            )

            # 推論ループはあえて sleep を入れず、センサ更新に合わせて動かしてもよい
            # 負荷が高ければ少しだけ sleep
            time.sleep(0.001)
            hand_img = None
            joint_state = None

    # ========== 4) スレッド起動 ==========
    pub_thread = threading.Thread(target=publish_loop, daemon=True)
    inf_thread = threading.Thread(target=inference_loop, daemon=True)

    pub_thread.start()
    inf_thread.start()

    # メインスレッドは待機（Ctrl+C で終了）
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt, exiting...")

if __name__ == "__main__":
    main()
