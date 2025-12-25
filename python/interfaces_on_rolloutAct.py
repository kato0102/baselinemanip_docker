# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("/opt/python")   # irsl_iceoryx2 用

from irsl_iceoryx2 import recvNumpy, sendNumpy
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
    rollout.image_transforms = v2.Compose(
        [v2.ToDtype(torch.float32, scale=True)]
    )

    state_meta = rollout.model_meta_info["state"]
    action_meta = rollout.model_meta_info["action"]

    state_dim = rollout.state_dim
    action_dim = rollout.action_dim

    print("[info] state_dim:", state_dim, "action_dim:", action_dim)

    state_keys = rollout.model_meta_info.get("state_keys", None)
    action_keys = rollout.model_meta_info.get("action_keys", None)
    print("[info] state_keys:", state_keys)
    print("[info] action_keys:", action_keys)

    # ========== 2) iceoryx インタフェース ==========
    sub_odom        = recvNumpy("ice_odom")          # (vx, vy, wz)
    sub_img         = recvNumpy("ice_image")         # head_rgbd
    sub_hand_img    = recvNumpy("ice_hand_image")    # hand camera
    sub_joint_state = recvNumpy("ice_joint_states")  # joint positions
    sub_head_state  = recvNumpy("ice_head_state")    # 未使用だが保持
    sub_arm_state   = recvNumpy("ice_arm_state")     # 未使用だが保持

    pub_twist       = sendNumpy("ice_twist")         # (vx, vy, wz)
    pub_arm_cmd     = sendNumpy("ice_arm_cmd_out")
    pub_head_cmd    = sendNumpy("ice_head_cmd_out")
    pub_gripper_cmd = sendNumpy("ice_gripper_cmd_out")

    print("[info] start ACT rollout over iceoryx")

    # ========== 3) 次元の定義 ==========
    JOINT_STATE_DIM = 8
    MOBILE_DIM      = 3

    JOINT_CMD_DIM   = 8   # command_joint_pos
    BASE_CMD_DIM    = 3   # command_mobile_omni_vel
    ACTION_DIM      = JOINT_CMD_DIM + BASE_CMD_DIM  # 11

    ARM_DIM     = 5   # arm_flex, arm_lift, arm_roll, wrist_flex, wrist_roll
    HEAD_DIM    = 2   # head_pan, head_tilt
    GRIPPER_DIM = 1   # hand_motor

    # ===== 共有変数（推論スレッドが書き込み、PUB スレッドが読み取り） =====
    cmd_lock = threading.Lock()
    latest_base_cmd    = np.zeros((BASE_CMD_DIM,), dtype=np.float64)
    latest_head_cmd    = np.zeros((HEAD_DIM,), dtype=np.float64)
    latest_arm_cmd     = np.zeros((ARM_DIM,), dtype=np.float64)
    latest_gripper_cmd = np.zeros((GRIPPER_DIM,), dtype=np.float64)
    has_action         = False  # まだ一度も推論していない間は publish しない

    # ===== Publish ループ（PUB_INTERVAL ごとに呼ばれる “コールバック”的役割）=====
    PUB_INTERVAL = 1/10  # ここで pub 周期だけを制御（10Hz）

    def publish_loop():
        nonlocal latest_base_cmd, latest_head_cmd, latest_arm_cmd, latest_gripper_cmd, has_action
        try:
            while True:
                time.sleep(PUB_INTERVAL)
                with cmd_lock:
                    if not has_action:
                        continue  # まだ初期推論が終わっていない
                    base_cmd  = latest_base_cmd.copy()
                    head_cmd  = latest_head_cmd.copy()
                    arm_cmd   = latest_arm_cmd.copy()
                    grip_cmd  = latest_gripper_cmd.copy()

                # ここで実際に pub（一定周期）
                pub_twist.sendAry(base_cmd)
                pub_head_cmd.sendAry(head_cmd)
                pub_arm_cmd.sendAry(arm_cmd)
                pub_gripper_cmd.sendAry(grip_cmd)

                # デバッグしたければここに print を置く
                #print("[pub] base:", base_cmd, "head:", head_cmd, "arm:", arm_cmd, "grip:", grip_cmd)

        except KeyboardInterrupt:
            print("[publish_loop] interrupted")

    # ===== 推論ループ（可能な限り回し続ける） =====
    def inference_loop():
        nonlocal latest_base_cmd, latest_head_cmd, latest_arm_cmd, latest_gripper_cmd, has_action

        while True:
            # --- 1) 最新のセンサ値を取得 ---
            odom        = sub_odom.getLastAry()
            img         = sub_img.getLastAry()
            hand_img    = sub_hand_img.getLastAry()
            joint_state = sub_joint_state.getLastAry()

            # odom / 画像 / joint_state のどれかが未到着なら待つ
            if joint_state is None or odom is None or img is None:
                time.sleep(0.005)
                continue

            js = np.asarray(joint_state, dtype=np.float32).reshape(-1)
            if js.shape[0] > JOINT_STATE_DIM:
                js = js[:JOINT_STATE_DIM]
            elif js.shape[0] < JOINT_STATE_DIM:
                js = np.pad(js, (0, JOINT_STATE_DIM - js.shape[0]))

            mv = np.asarray(odom, dtype=np.float32).reshape(-1)
            if mv.shape[0] > MOBILE_DIM:
                mv = mv[:MOBILE_DIM]
            elif mv.shape[0] < MOBILE_DIM:
                mv = np.pad(mv, (0, MOBILE_DIM - mv.shape[0]))

            state = np.concatenate([js, mv], axis=0)  # (12,)

            # ===== 画像（head + hand の2カメラ） =====
            np_head = np.asarray(img, dtype=np.uint8)

            if hand_img is not None:
                np_hand = np.asarray(hand_img, dtype=np.uint8)
            else:
                # hand カメラがまだ来ていないときの応急処置：headを二重に使う
                np_hand = np_head.copy()
            np_head = cv2.cvtColor(np_head, cv2.COLOR_BGR2RGB)
            np_hand = cv2.cvtColor(np_hand, cv2.COLOR_BGR2RGB)
            np_images = [np_head, np_hand]  # [head, hand]
            images = np_images

            # ===== ACT policy から action を計算（推論）=====
            action = rollout.step(state, images, do_plot=False)

            joint_cmd = action[:JOINT_CMD_DIM]            # (8,)
            base_cmd  = action[JOINT_CMD_DIM:ACTION_DIM]  # (3,)

            # joint_cmd = [arm_flex, arm_lift, arm_roll, wrist_flex, wrist_roll,
            #              head_pan, head_tilt, hand_motor]
            arm_cmd     = joint_cmd[0:5]   # arm_flex, arm_lift, arm_roll, wrist_flex, wrist_roll
            head_cmd    = joint_cmd[5:7]   # head_pan, head_tilt
            gripper_cmd = joint_cmd[7:8]   # hand_motor

            # ===== gripper の 0 / 1 / 中間処理 =====
            val = float(gripper_cmd[0])
            if val < 0.6:
                gripper_cmd = np.array([0.0], dtype=np.float64)
            elif val >= 0.8:
                gripper_cmd = np.array([1.0], dtype=np.float64)
            else:
                gripper_cmd = np.array([val], dtype=np.float64)

            # ===== 最新コマンドとして共有変数に書き込む（Publish スレッドが読む）=====
            with cmd_lock:
                latest_base_cmd    = np.asarray(base_cmd, dtype=np.float64)
                latest_head_cmd    = np.asarray(head_cmd, dtype=np.float64)
                latest_arm_cmd     = np.asarray(arm_cmd, dtype=np.float64)
                latest_gripper_cmd = np.asarray(gripper_cmd, dtype=np.float64)
                has_action         = True

            # デバッグ表示（推論側）
            print(
                "state(joint+odom):\n", state,
                " \n| base:", latest_base_cmd,
                " \n| head:", latest_head_cmd,
                " \n| arm:", latest_arm_cmd,
                "\n | grip:", latest_gripper_cmd,
            )

            # 推論ループはあえて sleep を入れず、センサ更新に合わせて動かしてもよい
            # 負荷が高ければ少しだけ sleep
            time.sleep(0.001)

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