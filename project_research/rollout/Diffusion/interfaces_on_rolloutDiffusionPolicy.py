#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import threading
import time

import cv2
import numpy as np

sys.path.append("/opt/python")
from irsl_manip_libs.irsl_iceoryx2 import recvNumpy, sendNumpy

from rollout_DiffusionPolicy import InteractiveRollout


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diffusion Policy rollout over iceoryx"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to Diffusion Policy checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=None,
        help="optional RoboManipBaselines rollout skip value",
    )
    parser.add_argument(
        "--drop_initial_actions",
        type=int,
        default=0,
        help=(
            "number of initial actions to drop from every predicted Diffusion "
            "Policy action chunk"
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show observation/action rollout plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # RolloutDiffusionPolicy parses sys.argv internally.
    rollout_argv = [
        "interfaces_on_rolloutDiffusionPolicy.py",
        "--checkpoint",
        args.checkpoint,
        "--drop_initial_actions",
        str(args.drop_initial_actions),
    ]
    if args.skip is not None:
        rollout_argv += ["--skip", str(args.skip)]
    if not args.plot:
        rollout_argv += ["--no_plot"]

    sys.argv = rollout_argv

    rollout = InteractiveRollout(use_plot=args.plot)

    print("[info] state_dim:", rollout.state_dim)
    print("[info] action_dim:", rollout.action_dim)
    print("[info] state_keys:", rollout.state_keys)
    print("[info] action_keys:", rollout.action_keys)
    print("[info] camera_names:", rollout.camera_names)
    print(
        "[info] n_obs_steps:",
        rollout.model_meta_info["data"]["n_obs_steps"],
    )
    print(
        "[info] n_action_steps:",
        rollout.model_meta_info["data"]["n_action_steps"],
    )

    # ===== iceoryx interface: same robot I/O as interfaces_on_rolloutAct.py =====
    sub_hand_img = recvNumpy("ice_hand_image")
    sub_joint_state = recvNumpy("ice_joint_states")

    pub_arm_cmd = sendNumpy("ice_arm_cmd_out")
    pub_gripper_cmd = sendNumpy("ice_gripper_cmd_out")

    print("[info] start Diffusion Policy rollout over iceoryx")

    JOINT_CMD_DIM = 7
    GRIPPER_CMD_DIM = 1
    EXPECTED_ACTION_DIM = JOINT_CMD_DIM + GRIPPER_CMD_DIM

    if rollout.action_dim != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            "This robot interface expects an 8-D action "
            "(7 arm joints + 1 gripper), but the checkpoint has "
            f"action_dim={rollout.action_dim}."
        )

    cmd_lock = threading.Lock()
    latest_arm_cmd = np.zeros((JOINT_CMD_DIM,), dtype=np.float64)
    latest_gripper_cmd = np.zeros((GRIPPER_CMD_DIM,), dtype=np.float64)
    has_action = False

    PUB_INTERVAL = 1.0 / 10.0

    def publish_loop():
        nonlocal has_action
        try:
            while True:
                time.sleep(PUB_INTERVAL)
                with cmd_lock:
                    if not has_action:
                        continue
                    arm_cmd = latest_arm_cmd.copy()
                    grip_cmd = latest_gripper_cmd.copy()
                    # Match the current ACT interface: publish each newly inferred
                    # action once rather than repeatedly replaying an old command.
                    has_action = False

                pub_arm_cmd.sendAry(arm_cmd)
                pub_gripper_cmd.sendAry(grip_cmd)
                print("[pub] arm:", arm_cmd, "grip:", grip_cmd)
        except KeyboardInterrupt:
            print("[publish_loop] interrupted")

    def inference_loop():
        nonlocal latest_arm_cmd, latest_gripper_cmd, has_action

        hand_img = None
        joint_state = None

        while True:
            # Diffusion Policy returns an action chunk.  Do not consume the next
            # element of that chunk until the previous command has actually been
            # published; otherwise a fast camera/joint-state stream could skip
            # several predicted actions before the 10 Hz publisher sees them.
            with cmd_lock:
                waiting_for_publish = has_action
            if waiting_for_publish:
                time.sleep(0.001)
                continue

            hand_img_latest = sub_hand_img.getLastAry()
            joint_state_latest = sub_joint_state.getLastAry()

            if hand_img_latest is not None:
                hand_img = hand_img_latest
            if joint_state_latest is not None:
                joint_state = joint_state_latest

            if joint_state is None or hand_img is None:
                time.sleep(0.005)
                continue

            state = np.asarray(joint_state, dtype=np.float32).reshape(-1)
            if state.shape[0] > rollout.state_dim:
                state = state[: rollout.state_dim]
            elif state.shape[0] < rollout.state_dim:
                state = np.pad(state, (0, rollout.state_dim - state.shape[0]))

            # Relay image is BGR; training/rollout observations are RGB.
            np_hand = np.asarray(hand_img, dtype=np.uint8)
            np_hand = cv2.cvtColor(np_hand, cv2.COLOR_BGR2RGB)

            if len(rollout.camera_names) != 1:
                raise RuntimeError(
                    "This interface currently receives one hand camera, but the "
                    f"checkpoint expects {len(rollout.camera_names)} cameras: "
                    f"{rollout.camera_names}"
                )

            images = [np_hand]

            action = rollout.step(state, images, do_plot=args.plot)

            arm_cmd = np.asarray(action[:JOINT_CMD_DIM], dtype=np.float64)
            gripper_cmd = np.asarray(action[JOINT_CMD_DIM:], dtype=np.float64)

            with cmd_lock:
                latest_arm_cmd = arm_cmd
                latest_gripper_cmd = gripper_cmd
                has_action = True

            print(
                "state:\n",
                state,
                "\n| arm:",
                latest_arm_cmd,
                "\n| grip:",
                latest_gripper_cmd,
            )

            # Consume each sensor sample once, as in the current ACT interface.
            hand_img = None
            joint_state = None
            time.sleep(0.001)

    pub_thread = threading.Thread(target=publish_loop, daemon=True)
    inf_thread = threading.Thread(target=inference_loop, daemon=True)

    pub_thread.start()
    inf_thread.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt, exiting...")


if __name__ == "__main__":
    main()
