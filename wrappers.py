from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_env import StepType, specs

from utils import segmentation_to_robot_mask


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats, use_metaworld_reward_dict=False):
        self._env = env
        self._num_repeats = num_repeats
        self.use_metaworld_reward_dict = use_metaworld_reward_dict

    def step(self, action):
        if self.use_metaworld_reward_dict:
            reward = 0.0
            success = False
            discount = 1.0
            for i in range(self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward["reward"] or 0.0) * discount
                success = success or time_step.reward["success"]
                discount *= time_step.discount
                if time_step.last():
                    break
            reward_dict = {"reward": reward, "success": success}
            return time_step._replace(reward=reward_dict, discount=discount)
        else:
            reward = 0.0
            discount = 1.0
            for i in range(self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward or 0.0) * discount
                discount *= time_step.discount
                if time_step.last():
                    break
            return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, frame_keys=["pixels"]):
        self._env = env
        self._num_frames = num_frames
        if not isinstance(frame_keys, list):
            frame_keys = [frame_keys]
        self._frame_keys = frame_keys
        self._frames = [deque([], maxlen=num_frames) for _ in range(len(frame_keys))]

        wrapped_obs_spec = env.observation_spec()
        for key in frame_keys:
            assert key in wrapped_obs_spec

            frame_shape = wrapped_obs_spec[key].shape
            frame_dtype = wrapped_obs_spec[key].dtype
            # remove batch dim
            if len(frame_shape) == 4:
                frame_shape = frame_shape[1:]
            wrapped_obs_spec[key] = specs.BoundedArray(
                shape=np.concatenate(
                    [[frame_shape[2] * num_frames], frame_shape[:2]], axis=0
                ),
                dtype=frame_dtype,
                minimum=0,
                maximum=255,
                name="observation",
            )
        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        for i, key in enumerate(self._frame_keys):
            assert len(self._frames[i]) == self._num_frames
            stacked_frames = np.concatenate(list(self._frames[i]), axis=0)
            obs[key] = stacked_frames
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step, key):
        pixels = time_step.observation[key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        for i, key in enumerate(self._frame_keys):
            pixels = self._extract_pixels(time_step, key)
            for _ in range(self._num_frames):
                self._frames[i].append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        for i, key in enumerate(self._frame_keys):
            pixels = self._extract_pixels(time_step, key)
            self._frames[i].append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env, using_metaworld=False):
        self._env = env
        self.using_metaworld = using_metaworld

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        if time_step.reward is None and self.using_metaworld:
            reward = {"reward": 0.0, "success": 0}
        elif time_step.reward is None:
            reward = 0.0
        else:
            reward = time_step.reward
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class SegmentationToRobotMaskWrapper(dm_env.Environment):
    def __init__(self, env, segmentation_key="segmentation"):
        self._env = env
        self.segmentation_key = segmentation_key
        assert segmentation_key in env.observation_spec()

        wrapped_obs_spec = env.observation_spec()
        frame_shape = wrapped_obs_spec[segmentation_key].shape
        frame_dtype = wrapped_obs_spec[segmentation_key].dtype
        # remove batch dim
        if len(frame_shape) == 4:
            frame_shape = frame_shape[1:]
        wrapped_obs_spec[segmentation_key] = specs.BoundedArray(
            shape=np.concatenate([frame_shape[:2], [1]], axis=0),
            dtype=frame_dtype,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._obs_spec = wrapped_obs_spec

    def _transform_observation(self, time_step):
        obs = time_step.observation
        seg = obs[self.segmentation_key]

        robot_mask = segmentation_to_robot_mask(seg)
        robot_mask = robot_mask.astype(seg.dtype)
        robot_mask = robot_mask.reshape(robot_mask.shape[0], robot_mask.shape[1], 1)

        obs[self.segmentation_key] = robot_mask
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
