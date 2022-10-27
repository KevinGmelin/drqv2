# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np
import torch
import wandb


class VideoRecorder:
    def __init__(self, root_dir, metaworld_camera_name=None, render_size=256, fps=20, use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.metaworld_camera_name = metaworld_camera_name
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, "mt1"):
                frame = env.physics.render(
                    height=self.render_size, width=self.render_size,
                    mode='offscreen',
                    camera_name=self.metaworld_camera_name,
                )
            elif hasattr(env, "physics"):
                frame = env.physics.render(
                    height=self.render_size, width=self.render_size, camera_id=0
                )
            else:
                frame = env.render()
            self.frames.append(frame)

    def save(self, file_name, step):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            if self.use_wandb:
                wandb.log({"eval_video": wandb.Video(str(path), fps=self.fps, format="mp4")}, step=step)


class ReconstructionRecorder:
    def __init__(self, root_dir, encoder, decoder, device, metaworld_camera_name=None, render_size=84, fps=20, use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.metaworld_camera_name = metaworld_camera_name
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, "mt1"):
                frame = env.physics.render(
                    height=self.render_size, width=self.render_size,
                    mode='offscreen',
                    camera_name=self.metaworld_camera_name,
                )
            elif hasattr(env, "physics"):
                frame = env.physics.render(
                    height=self.render_size, width=self.render_size, camera_id=0
                )
            else:
                frame = env.render()
            
            # Encoder takes a stack of frames - duplicate the current frame for the purpose of this recording
            frame = np.tile(frame, self.encoder.obs_shape[0]//frame.shape[2])
            frame = torch.FloatTensor(np.transpose(frame, (2, 0, 1))).to(self.device)
            frame = self.decoder(self.encoder(frame.unsqueeze(0))).detach().cpu().numpy()
            frame = np.transpose(frame[0, 0:3, :, :], (1,2,0))
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            self.frames.append(frame)

    def save(self, file_name, step):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            if self.use_wandb:
                wandb.log({"reconstruction_video": wandb.Video(str(path), fps=self.fps, format="mp4")}, step=step)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / "train_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(
                obs[-3:].transpose(1, 2, 0),
                dsize=(self.render_size, self.render_size),
                interpolation=cv2.INTER_CUBIC,
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
