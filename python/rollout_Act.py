from robo_manip_baselines.policy.act import RolloutAct
import torch
from torchvision.transforms import v2
from robo_manip_baselines.common.utils.DataUtils import normalize_data
from robo_manip_baselines.common import denormalize_data

from robo_manip_baselines.policy.act import RolloutAct
import torch
from torchvision.transforms import v2
import cv2
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
class InteractiveRollout(RolloutAct):
    """
    学習済み ACT ポリシーをロードして、state + image から action を出すためのクラス。
    実際の checkpoint 指定は、呼び出し側（interfaces_on_rollout.py）で sys.argv に
    --checkpoint を入れて行う。
    """

    def __init__(self):
        self.policy_name = "Act"
        self.setup_args()              # argparse で --checkpoint などを読む
        # self.setup_env(render_mode=render_mode)  # 環境は今回使わないので無効
        self.setup_model_meta_info()   # model_meta_info.pkl をロード
        self.setup_policy()            # ACT Policy 構築 & 重みロード
        self.setup_plot()
        self.reset_variables()   # policy_action_list の初期化など
        # デフォルトの画像前処理（uint8 → float32[0,1]）
        self.rollout_time_idx = 0
        self.image_transforms = v2.Compose(
            [v2.ToDtype(torch.float32, scale=True)]
        )
    def setup_plot(self):
        # 画面を持たないバックエンド
        matplotlib.use("agg")

        # RolloutAct と同じ 2 行 × N 列構成
        n_cols = max(
            len(self.camera_names) + 1,  # カメラ枚数 + action 用
            len(self.policy.model.transformer.encoder.layers),  # attention 用
        )
        self.fig, self.ax = plt.subplots(
            2,
            n_cols,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )

        # 軸の初期化
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # キャンバス作成 & 空の図を一度描画して OpenCV で表示
        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(
                np.asarray(self.canvas.buffer_rgba()),
                cv2.COLOR_RGB2BGR,
            ),
        )
        cv2.waitKey(1)

        # env が無いので DataKey.get_plot_scale は使わず、全部スケール 1 にする
        if self.action_dim > 0:
            self.action_plot_scale = np.ones(self.action_dim, dtype=np.float32)
        else:
            self.action_plot_scale = np.zeros(0, dtype=np.float32)

    # ---------------------------------------------------------
    # ★ infer_policy で使うバッファの初期化（RolloutAct の reset_variables + α）
    # ---------------------------------------------------------
    def reset_variables(self):
        # RolloutBase.reset_variables() 相当（policy_action_list の初期化）
        self.policy_action_list = np.empty((0, self.action_dim))

        # RolloutAct で使っていたバッファもここで初期化
        self.policy_action_buf = []
        self.policy_action_buf_history = []

    # ---------------------------------------------------------
    # ★ 外部から state / images をセットするメソッド
    # ---------------------------------------------------------
    def set_input(self, state_np, images_np):
        """
        state_np : shape = (state_dim,)
        images_np: shape = (num_cameras, H, W, 3) の uint8 or float
        """
        self._input_state = np.asarray(state_np, dtype=np.float32)
        # 画像は uint8 前提に合わせる（0-255）
        self._input_images = np.asarray(images_np, dtype=np.uint8)

        # draw_plot() / plot_images() が使う info["rgb_images"] をセット
        self.info = {
            "rgb_images": {
                name: self._input_images[i]
                for i, name in enumerate(self.camera_names)
            }
        }


    # ---------------------------------------------------------
    # ★ env 依存を消した get_state / get_images のオーバーライド
    #    → infer_policy() から呼ばれる
    # ---------------------------------------------------------
    def get_state(self):
        """
        RolloutBase.get_state を上書き。
        env & MotionManager からではなく、set_input でセットした値から作る。
        """
        if not hasattr(self, "_input_state"):
            raise RuntimeError("set_input() で state をセットしてから infer_policy() を呼んでください。")

        state = self._input_state
        # 学習時と同じ正規化
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state[np.newaxis], dtype=torch.float32).to(self.device)
        return state

    def get_images(self):
        """
        RolloutBase.get_images を上書き。
        env.info からではなく、set_input で渡した images から作る。
        """
        if not hasattr(self, "_input_images"):
            raise RuntimeError("set_input() で images をセットしてから infer_policy() を呼んでください。")

        images = self._input_images  # (Cameras, H, W, 3)

        # (N, H, W, C) → (N, C, H, W)
        images = np.moveaxis(images, -1, -3)
        images = torch.tensor(images, dtype=torch.uint8)
        images = self.image_transforms(images)[torch.newaxis].to(self.device)
        return images
    def draw_plot(self):
        # まずは一旦全部クリア
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # --- 1) 画像を描画 ---
        # self.info["rgb_images"][camera_name] を使う
        # set_input() の中で self.info を設定している前提
        if hasattr(self, "info") and "rgb_images" in self.info:
            # 1行目の 0〜(カメラ数-1) を画像用に使う
            self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # --- 2) action を描画 ---
        # 1行目のカメラの次の列を action プロットに使う
        if self.policy_action_list.shape[0] > 0:
            self.plot_action(self.ax[0, len(self.camera_names)])

        # --- 3) attention 画像 ---
        # attention_shape は元コードに合わせて (15, 20 * camera_num)
        attention_shape = (15, 20 * len(self.camera_names)) if len(self.camera_names) > 0 else (15, 20)

        for layer_idx, layer in enumerate(self.policy.model.transformer.encoder.layers):
            if layer_idx >= self.ax.shape[1]:
                # 念のため、列数を超えそうなら無視
                break

            if getattr(layer.self_attn, "correlation_mat", None) is None:
                continue

            # correlation_mat[2:, 1] を reshape してヒートマップとして描画
            corr = layer.self_attn.correlation_mat
            try:
                att_image = corr[2:, 1].reshape(attention_shape)
            except ValueError:
                # reshape できない場合はスキップ（トークン数が合わないなど）
                continue

            self.ax[1, layer_idx].imshow(att_image)
            self.ax[1, layer_idx].set_title(
                f"attention image ({layer_idx})", fontsize=20
            )

        # --- 4) 最後にキャンバスを描画して OpenCV で表示 ---
        self.canvas.draw()
        img = np.asarray(self.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.policy_name, img_bgr)
        cv2.waitKey(1)
        # ---------------------------------------------------------
    # ★ infer_policy() は RolloutAct の実装をそのまま利用
    #   （ここではオーバーライドしない）
    #
    #   呼ぶと:
    #    - self.policy_action (denormalized) が更新される
    #    - self.policy_action_list に履歴が追加される
    #    - temporal ensembling も有効なら適用される
    # ---------------------------------------------------------

    # ついでに：1 ステップ分だけ推論 & 可視化するヘルパー
    

    def step(self, state_np, images_np, do_plot=True):
        self.set_input(state_np, images_np)

        t0 = time.time()
          # ★ 後で説明
        self.infer_policy()
        t1 = time.time()

        if do_plot and (not getattr(self.args, "no_plot", False)):
            self.draw_plot()
        t2 = time.time()

        print(f"[step] infer: {t1 - t0:.3f}s, plot: {t2 - t1:.3f}s")

        self.rollout_time_idx += 1
        return self.policy_action