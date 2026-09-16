import time

import cv2
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision.transforms import v2

from robo_manip_baselines.common import normalize_data
from robo_manip_baselines.policy.diffusion_policy import RolloutDiffusionPolicy


class InteractiveRollout(RolloutDiffusionPolicy):
    """
    Environment-independent Diffusion Policy rollout.

    The original RolloutDiffusionPolicy reads observations from a Gym environment
    and MotionManager.  This subclass accepts state/image observations supplied by
    the iceoryx interface while keeping RoboManipBaselines' observation history and
    action-chunk inference logic intact.
    """

    def __init__(self, use_plot=False):
        self.policy_name = "DiffusionPolicy"

        # Parse --checkpoint, --skip, --drop_initial_actions, etc.
        self.setup_args()

        # Do not create a Gym environment here.
        self.setup_model_meta_info()
        self.setup_policy()

        # RolloutBase.__init__ is intentionally bypassed, so initialize the
        # preprocessing transform here explicitly.
        self.image_transforms = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True)]
        )

        if use_plot:
            self.setup_plot()

        self.reset_variables()
        self.rollout_time_idx = 0

    def setup_plot(self):
        """Create a plot without accessing self.env."""
        matplotlib.use("agg")

        if len(self.camera_names) > 0:
            num_plot_rows = 2
            num_plot_cols = max(len(self.camera_names), 1)
        else:
            num_plot_rows = 1
            num_plot_cols = 1

        self.fig, self.ax = plt.subplots(
            num_plot_rows,
            num_plot_cols,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(
                np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR
            ),
        )
        cv2.waitKey(1)

        # DataKey.get_plot_scale_for_policy() requires an environment.
        self.action_plot_scale = np.ones(self.action_dim, dtype=np.float32)

    def reset_variables(self):
        """Initialize the buffers used by RolloutDiffusionPolicy.infer_policy()."""
        self.policy_action_list = np.empty((0, self.action_dim))
        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def set_input(self, state_np, images_np):
        """
        Set the newest external observation.

        Args:
            state_np: shape (state_dim,)
            images_np: sequence/array with shape
                       (num_cameras, H, W, 3), RGB uint8.
        """
        state = np.asarray(state_np, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"state dimension mismatch: got {state.shape[0]}, "
                f"expected {self.state_dim}"
            )
        self._input_state = state

        if len(self.camera_names) > 0:
            images = np.asarray(images_np, dtype=np.uint8)
            if images.ndim != 4:
                raise ValueError(
                    "images_np must have shape (num_cameras, H, W, 3); "
                    f"got {images.shape}"
                )
            if images.shape[0] != len(self.camera_names):
                raise ValueError(
                    f"camera count mismatch: got {images.shape[0]}, "
                    f"expected {len(self.camera_names)} ({self.camera_names})"
                )
            if images.shape[-1] != 3:
                raise ValueError(
                    f"images must have 3 channels; got shape {images.shape}"
                )

            self._input_images = images
            self.info = {
                "rgb_images": {
                    name: self._input_images[i]
                    for i, name in enumerate(self.camera_names)
                }
            }
        else:
            self._input_images = np.empty((0,), dtype=np.uint8)
            self.info = {"rgb_images": {}}

    def update_state_buf(self):
        """
        External-input replacement for RolloutDiffusionPolicy.update_state_buf().

        Diffusion Policy consumes n_obs_steps observations.  At the first step,
        duplicate the first observation to fill the history, matching the original
        RoboManipBaselines behavior.
        """
        if not hasattr(self, "_input_state"):
            raise RuntimeError(
                "Call set_input() before infer_policy()/step()."
            )

        state = normalize_data(self._input_state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        n_obs_steps = self.model_meta_info["data"]["n_obs_steps"]
        if self.state_buf is None:
            self.state_buf = [state.clone() for _ in range(n_obs_steps)]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

    # update_images_buf() and get_images() from RolloutDiffusionPolicy can be
    # reused because set_input() populates self.info["rgb_images"].  They perform
    # exactly the resize, CHW conversion, [0,1] scaling, and [-1,1] adjustment
    # used by the original rollout implementation.

    def draw_plot(self):
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        if len(self.camera_names) > 0:
            self.plot_images(self.ax[0, 0 : len(self.camera_names)])
            if self.policy_action_list.shape[0] > 0:
                self.plot_action(self.ax[1, 0])
        elif self.policy_action_list.shape[0] > 0:
            self.plot_action(self.ax[0, 0])

        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(
                np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR
            ),
        )
        cv2.waitKey(1)

    def step(self, state_np, images_np, do_plot=True):
        """Run one control step and return one denormalized action."""
        self.set_input(state_np, images_np)

        t0 = time.time()
        # The standard RolloutPhase runs policy inference under inference_mode().
        # Do the same here to avoid autograd graph allocation during deployment.
        with torch.inference_mode():
            self.infer_policy()
        t1 = time.time()

        if do_plot and (not getattr(self.args, "no_plot", False)):
            self.draw_plot()
        t2 = time.time()

        print(f"[step] infer: {t1 - t0:.3f}s, plot: {t2 - t1:.3f}s")

        self.rollout_time_idx += 1
        return self.policy_action
