from robo_manip_baselines.policy.act import RolloutAct
import torch
from torchvision.transforms import v2
from robo_manip_baselines.common.utils.DataUtils import normalize_data
from robo_manip_baselines.common import denormalize_data

class InteractiveRollout(RolloutAct):
    def __init__(self):
        self.policy_name = 'Act'
        self.setup_args()
        # self.setup_env(render_mode=render_mode)
        self.setup_model_meta_info()
        self.setup_policy()

import numpy as np
import sys
sys.argv.append('--checkpoint')
#sys.argv.append('/userdir/new00_Act_20251106_121251/policy_best.ckpt')
sys.argv.append('/userdir/test_hsr/checkpoint/policy_best.ckpt')

rollout = InteractiveRollout()
rollout.image_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

### functions
np_state  = np.zeros(3) ## TODO should be add
np_images = [ numpy.array ] ## TODO should be add (images)

state = normalize_data(np_state, rollout.model_meta_info["state"])
state = torch.tensor(state[np.newaxis], dtype=torch.float32).to(rollout.device)

images = np.stack(
            np_images,
            axis=0,
        )
print("raw_image: ", images)
images = np.moveaxis(images, -1, -3)
images = torch.tensor(images, dtype=torch.uint8)
images = rollout.image_transforms(images)[torch.newaxis].to(rollout.device)

actions = rollout.policy(state, images) ##

np_action = actions[0][0].cpu().detach().numpy().astype(np.float64)
norm_action = denormalize_data(np_action, rollout.model_meta_info["action"])

