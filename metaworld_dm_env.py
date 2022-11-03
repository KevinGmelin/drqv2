import metaworld

import dm_env
from dm_env import specs
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from wrappers import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    FrameStackWrapper,
    ExtendedTimeStepWrapper,
)
import random
import gym


class Render_Wrapper:
    def __init__(self, render_fn):
        self.render_fn = render_fn

    def render(self, *args, **kwargs):
        return self.render_fn(*args, **kwargs)


class MT1_Wrapper(dm_env.Environment):
    def __init__(
        self, env_name: str, discount=1.0, seed=None, proprioceptive_state=True
    ):
        self.env_name = env_name
        self.discount = discount
        self.mt1 = metaworld.MT1(env_name, seed=seed)
        self._env = self.mt1.train_classes[env_name]()
        self.physics = Render_Wrapper(self._env.sim.render)
        self._reset_next_step = True
        self.current_step = 0
        self.proprioceptive_state = proprioceptive_state
        self.NUM_PROPRIOCEPTIVE_STATES = 7

        assert isinstance(self._env.observation_space, gym.spaces.Box)
        assert isinstance(self._env.action_space, gym.spaces.Box)

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.current_step = 0
        task = random.choice(self.mt1.train_tasks)
        self._env.set_task(task)
        observation = self._env.reset()
        if self.proprioceptive_state:
            observation = self.get_proprioceptive_observation(observation)
        observation = observation.astype(self._env.observation_space.dtype)
        self.current_step += 1
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observation, reward, _, info = self._env.step(action)
        self.current_step += 1

        if self.proprioceptive_state:
            observation = self.get_proprioceptive_observation(observation)

        observation = observation.astype(self._env.observation_space.dtype)

        reward_dict = {"reward": reward, "success": info["success"]}

        if self.current_step == self._env.max_path_length:
            self._reset_next_step = True
            return dm_env.truncation(reward_dict, observation, self.discount)

        return dm_env.transition(reward_dict, observation, self.discount)

    def get_proprioceptive_observation(self, observation):
        observation = observation[0 : self.NUM_PROPRIOCEPTIVE_STATES]
        return observation

    def observation_spec(self) -> specs.BoundedArray:
        if self.proprioceptive_state:
            return specs.BoundedArray(
                shape=(self.NUM_PROPRIOCEPTIVE_STATES,),
                dtype=self._env.observation_space.dtype,
                minimum=self._env.observation_space.low[
                    0 : self.NUM_PROPRIOCEPTIVE_STATES
                ],
                maximum=self._env.observation_space.high[
                    0 : self.NUM_PROPRIOCEPTIVE_STATES
                ],
                name="observation",
            )
        else:
            return specs.BoundedArray(
                shape=self._env.observation_space.shape,
                dtype=self._env.observation_space.dtype,
                minimum=self._env.observation_space.low,
                maximum=self._env.observation_space.high,
                name="observation",
            )

    def action_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=self._env.action_space.shape,
            dtype=self._env.action_space.dtype,
            minimum=self._env.action_space.low,
            maximum=self._env.action_space.high,
            name="action",
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_metaworld(
    name,
    frame_stack,
    action_repeat,
    discount,
    seed,
    camera_name,
    add_segmentation_to_obs,
):
    env = MT1_Wrapper(
        env_name=name, discount=discount, seed=seed, proprioceptive_state=True
    )

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat, use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys = []

    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    render_kwargs = dict(height=84, width=84, mode="offscreen", camera_name=camera_name)
    env = pixels.Wrapper(
        env, pixels_only=False, render_kwargs=render_kwargs, observation_key=rgb_key
    )

    if add_segmentation_to_obs:
        segmentation_key = "segmentation"
        frame_keys.append(segmentation_key)
        segmentation_kwargs = dict(
            height=84,
            width=84,
            mode="offscreen",
            camera_name=camera_name,
            segmentation=True,
        )
        env = pixels.Wrapper(
            env,
            pixels_only=False,
            render_kwargs=segmentation_kwargs,
            observation_key=segmentation_key,
        )

    env = FrameStackWrapper(env, frame_stack, frame_keys)
    env = ExtendedTimeStepWrapper(env, using_metaworld=True)
    return env
