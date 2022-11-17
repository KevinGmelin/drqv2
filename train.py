# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
from metaworld_dm_env import make_metaworld
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder, ReconstructionRecorder
from video import MaskRecorder, ReconstructedMaskRecorder

import wandb
from omegaconf import OmegaConf
from collections import OrderedDict

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    assert (
        "pixels" in obs_spec
    ), "Observation spec passed to make_agent must contain a observation named 'pixels'"
    cfg.obs_shape = obs_spec["pixels"].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(
            self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb
        )
        # create envs
        if self.cfg.using_metaworld:
            self.train_env = make_metaworld(
                self.cfg.task_name.split("_")[1],
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.discount,
                self.cfg.seed,
                self.cfg.camera_name,
                add_segmentation_to_obs=(self.cfg.agent.disentangled_version > 0),
            )
            self.eval_env = make_metaworld(
                self.cfg.task_name.split("_")[1],
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.discount,
                self.cfg.seed,
                self.cfg.camera_name,
                add_segmentation_to_obs=(self.cfg.agent.disentangled_version > 0),
            )
            reward_spec = OrderedDict(
                [
                    ("reward", specs.Array((1,), np.float32, "reward")),
                    ("success", specs.Array((1,), np.int16, "reward")),
                ]
            )
        else:
            self.train_env = dmc.make(
                self.cfg.task_name,
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.seed,
            )
            self.eval_env = dmc.make(
                self.cfg.task_name,
                self.cfg.frame_stack,
                self.cfg.action_repeat,
                self.cfg.seed,
            )
            reward_spec = specs.Array((1,), np.float32, "reward")

        discount_spec = specs.Array((1,), np.float32, "discount")
        data_specs = {
            "observation": self.train_env.observation_spec(),
            "action": self.train_env.action_spec(),
            "reward": reward_spec,
            "discount": discount_spec,
        }
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
        )
        self._replay_iter = None

        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.agent,
        )

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            self.cfg.camera_name,
            use_wandb=self.cfg.use_wandb,
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

        if self.cfg.save_reconstruction_video:
            assert (
                self.cfg.agent.use_decoder
                or self.cfg.agent.disentangled_version == 1
                or self.cfg.agent.disentangled_version == 3
            ), "Attempting to save a reconstruction video, but use_decoder is set to false."
            use_first_half_latent = (
                self.cfg.agent.disentangled_version == 1
                or self.cfg.agent.disentangled_version == 3
            )
            self.recon_recorder = ReconstructionRecorder(
                self.work_dir,
                self.agent.encoder,
                self.agent.decoder,
                self.agent.device,
                self.cfg.camera_name,
                use_wandb=self.cfg.use_wandb,
                use_first_half_latent=use_first_half_latent,
            )
        else:
            self.recon_recorder = ReconstructionRecorder(None, None, None, None, None)

        self.mask_recorder = MaskRecorder(
            self.work_dir if self.cfg.save_mask_video else None,
            metaworld_camera_name=self.cfg.camera_name,
            use_wandb=self.cfg.use_wandb,
        )

        if self.cfg.save_reconstructed_mask_video:
            assert (
                self.cfg.agent.disentangled_version > 0
            ), "Saving reconstructed mask video currently only supported for disentangled version > 0"
            if (
                self.cfg.agent.disentangled_version == 1
                or self.cfg.agent.disentangled_version == 3
            ):
                mask_decoder = self.agent.mask_decoder
            else:
                mask_decoder = self.agent.robot_mask_decoder
            self.recon_mask_recorder = ReconstructedMaskRecorder(
                self.work_dir,
                self.agent.encoder,
                mask_decoder,
                self.agent.device,
                self.cfg.camera_name,
                use_wandb=self.cfg.use_wandb,
                use_second_half_latent=True,
            )
        else:
            self.recon_mask_recorder = ReconstructedMaskRecorder(
                None, None, None, None, None
            )

        if self.cfg.use_wandb:
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            wandb.init(
                project=self.cfg.wandb.project_name,
                config=cfg_dict,
                name=self.cfg.wandb.run_name,
            )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        if self.cfg.using_metaworld:
            mean_max_success, mean_mean_success, mean_last_success = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            if self.cfg.using_metaworld:
                current_episode_max_success = 0
                current_episode_mean_success = 0
                current_episode_last_success = 0
            current_episode_step = 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            self.recon_recorder.init(self.eval_env, enabled=(episode == 0))
            self.mask_recorder.init(self.eval_env, enabled=(episode == 0))
            self.recon_mask_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                current_episode_step += 1
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                self.recon_recorder.record(self.eval_env)
                self.mask_recorder.record(self.eval_env)
                self.recon_mask_recorder.record(self.eval_env)
                if self.cfg.using_metaworld:
                    total_reward += time_step.reward["reward"]
                    success = int(time_step.reward["success"])
                    current_episode_max_success = max(
                        current_episode_max_success, success
                    )
                    current_episode_last_success = success
                    current_episode_mean_success += success
                else:
                    total_reward += time_step.reward
                step += 1
            mean_max_success += current_episode_max_success
            mean_last_success += current_episode_last_success
            mean_mean_success += current_episode_mean_success / current_episode_step
            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4", step=self.global_frame)
            self.recon_recorder.save(
                f"{self.global_frame}_decoder.mp4", step=self.global_frame
            )
            self.mask_recorder.save(
                f"{self.global_frame}_mask.mp4", step=self.global_frame
            )
            self.recon_mask_recorder.save(
                f"{self.global_frame}_mask_decoder.mp4", step=self.global_frame
            )

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            if self.cfg.using_metaworld:
                log("max_success", mean_max_success / episode)
                log("last_success", mean_last_success / episode)
                log("mean_success", mean_mean_success / episode)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        if self.cfg.using_metaworld:
            mean_success = 0
            max_success = 0
            last_success = 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)
                        if self.cfg.using_metaworld:
                            log("mean_success", mean_success / episode_step)
                            log("max_success", max_success)
                            log("last_success", last_success)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0
                if self.cfg.using_metaworld:
                    mean_success = 0
                    max_success = 0
                    last_success = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation, self.global_step, eval_mode=False
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action)
            if self.cfg.using_metaworld:
                episode_reward += time_step.reward["reward"]
                success = int(time_step.reward["success"])
                max_success = max(max_success, success)
                last_success = success
                mean_success += success
            else:
                episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
