# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),  # 41 x 41
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 39 x 39
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 37 x 37
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),  # 35 x 35
            nn.ReLU(),
        )

        # self.repr_dim = 256*3*3

        # self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 4, stride=2), # 32 x 41 x 41
        #                              nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), # 64 x 19 x 19
        #                              nn.ReLU(), nn.Conv2d(64, 128, 4, stride=2), # 128 x 8 x 8
        #                              nn.ReLU(), nn.Conv2d(128, 256, 4, stride=2), # 256 x 3 x 3
        #                              nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_act=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_act = output_act

        self.transpose_convnet = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, 3, 1),  # 37 x 37
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1),  # 39 x 39
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, 1),  # 41 x 41
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 3, 2, 0, 1),
        )  # 84 x 84
        # self.transpose_convnet = nn.Sequential(
        #                         nn.ConvTranspose2d(256, 128, 4, 2), # 8 x 8
        #                         nn.ReLU(),
        #                         nn.ConvTranspose2d(128, 64, 4, 2, 0, 1),  # 19 x 19
        #                         nn.ReLU(),
        #                         nn.ConvTranspose2d(64, 32, 4, 2, 0, 1),  # 41 x 41
        #                         nn.ReLU(),
        #                         nn.ConvTranspose2d(32, obs_shape[0], 4, 2)) # 84 x 84

        self.apply(utils.weight_init)

    def forward(self, x):
        out = x.view(x.shape[0], self.in_channels, 35, 35)
        out = self.transpose_convnet(out)
        out = (out + 0.5) * 255.0
        if self.output_act is not None:
            out = self.output_act(out)
        return out


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        use_decoder=False,
        reconstruction_loss_coeff=0.0,
        backprop_decoder_loss_to_encoder=False,
        decoder_lr=None,
        disentangled_version=-1,
        mask_lr=None,
        mask_loss_coeff=0.0,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_decoder = use_decoder
        self.reconstruction_loss_coeff = reconstruction_loss_coeff
        self.backprop_decoder_loss_to_encoder = backprop_decoder_loss_to_encoder
        self.disentangled_version = disentangled_version
        self.mask_loss_coef = mask_loss_coeff

        # models
        self.encoder = Encoder(obs_shape).to(device)
        if disentangled_version == 1 or disentangled_version == 3:
            self.mask_decoder = Decoder(
                in_channels=16, out_channels=obs_shape[0] // 3, output_act=nn.Sigmoid()
            ).to(device)
            self.decoder = Decoder(in_channels=16, out_channels=obs_shape[0]).to(device)
        elif disentangled_version == 2:
            self.robot_mask_decoder = Decoder(
                in_channels=16, out_channels=obs_shape[0] // 3, output_act=nn.Sigmoid()
            ).to(device)
            self.non_robot_mask_decoder = Decoder(
                in_channels=16, out_channels=obs_shape[0] // 3, output_act=nn.Sigmoid()
            ).to(device)
        elif use_decoder:
            self.decoder = Decoder(in_channels=32, out_channels=obs_shape[0]).to(device)

        self.actor = Actor(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)

        self.critic = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Loss functions
        if disentangled_version == 1 or disentangled_version == 3:
            self.reconstruction_loss_fn = nn.MSELoss(reduction="none")
            self.mask_loss_fn = nn.BCELoss(reduction="none")
        elif disentangled_version == 2:
            self.non_robot_mask_loss_fn = nn.BCELoss(reduction="none")
            self.robot_mask_loss_fn = nn.BCELoss(reduction="none")
        elif use_decoder:
            self.reconstruction_loss_fn = nn.MSELoss(reduction="none")

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if disentangled_version == 1 or disentangled_version == 3:
            assert (
                decoder_lr is not None
            ), "Decoder lr must be set for disentangled versions 1 and 3"
            assert (
                mask_lr is not None
            ), "Mask lr must be set for disentangled versions 1 and 3"
            self.decoder_opt = torch.optim.Adam(
                self.decoder.parameters(), lr=decoder_lr
            )
            self.mask_opt = torch.optim.Adam(self.mask_decoder.parameters(), lr=mask_lr)
        elif disentangled_version == 2:
            assert mask_lr is not None, "Mask lr must be set for disentangled version 2"
            self.robot_mask_opt = torch.optim.Adam(
                self.robot_mask_decoder.parameters(), lr=mask_lr
            )
            self.non_robot_mask_opt = torch.optim.Adam(
                self.non_robot_mask_decoder.parameters(), lr=mask_lr
            )
        elif use_decoder:
            assert (
                decoder_lr is not None
            ), "Decoder lr must be set if use_decoder is true"
            self.decoder_opt = torch.optim.Adam(
                self.decoder.parameters(), lr=decoder_lr
            )

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        if self.disentangled_version == 1 or self.disentangled_version == 3:
            self.decoder.train(training)
            self.mask_decoder.train(training)
        elif self.disentangled_version == 2:
            self.robot_mask_decoder.train(training)
            self.non_robot_mask_decoder.train(training)
        elif self.use_decoder:
            self.decoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = obs["pixels"]
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    # Critic and decoder are coupled since, in the case where backprop_decoder_loss_to_encoder is true, then
    # both the critic and decoder losses go through the encoder
    def update_critic_and_decoder(
        self,
        obs,
        encoded_obs,
        action,
        reward,
        discount,
        encoded_next_obs,
        step,
        robot_masks=None,
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(encoded_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(encoded_next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(encoded_obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.disentangled_version == 1 or self.disentangled_version == 3:
            f1 = encoded_obs[:, : int(encoded_obs.shape[1] / 2)]
            f2 = encoded_obs[:, int(encoded_obs.shape[1] / 2) :]
            reconstructed_obs = self.decoder(f1)
            reconstructed_mask = self.mask_decoder(f2)

            obs = obs / 255.0 - 0.5
            reconstructed_obs = torch.clamp(reconstructed_obs / 255.0 - 0.5, -0.5, 0.5)

            reconstruction_loss = self.reconstruction_loss_fn(reconstructed_obs, obs)
            if self.disentangled_version == 3:
                reconstruction_loss = torch.where(
                    torch.repeat_interleave(robot_masks, 3, dim=1) > 0,
                    0,
                    reconstruction_loss,
                )
            reconstruction_loss = (
                reconstruction_loss.reshape(obs.shape[0], -1).sum(dim=1).mean()
            )

            mask_loss = self.mask_loss_fn(reconstructed_mask, robot_masks)
            mask_loss = mask_loss.reshape(robot_masks.shape[0], -1).sum(dim=1).mean()

            loss = (
                critic_loss
                + reconstruction_loss * self.reconstruction_loss_coeff
                + mask_loss * self.mask_loss_coef
            )

            self.encoder_opt.zero_grad(set_to_none=True)
            self.decoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.mask_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()
            self.decoder_opt.step()
            self.mask_opt.step()
        elif self.disentangled_version == 2:
            f1 = encoded_obs[:, : int(encoded_obs.shape[1] / 2)]
            f2 = encoded_obs[:, int(encoded_obs.shape[1] / 2) :]
            reconstructed_non_robot_mask = self.non_robot_mask_decoder(f1)
            reconstructed_robot_mask = self.robot_mask_decoder(f2)

            robot_mask_loss = self.robot_mask_loss_fn(
                reconstructed_robot_mask, robot_masks
            )
            robot_mask_loss = (
                robot_mask_loss.reshape(robot_masks.shape[0], -1).sum(dim=1).mean()
            )

            non_robot_mask_loss = self.non_robot_mask_loss_fn(
                reconstructed_non_robot_mask, 1 - robot_masks
            )
            non_robot_mask_loss = (
                non_robot_mask_loss.reshape(robot_masks.shape[0], -1).sum(dim=1).mean()
            )

            loss = (
                critic_loss
                + non_robot_mask_loss * self.mask_loss_coef
                + robot_mask_loss * self.mask_loss_coef
            )

            self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.robot_mask_opt.zero_grad(set_to_none=True)
            self.non_robot_mask_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()
            self.robot_mask_opt.step()
            self.non_robot_mask_opt.step()
        elif self.use_decoder and self.backprop_decoder_loss_to_encoder:
            reconstructed_obs = self.decoder(encoded_obs)
            obs = obs / 255.0 - 0.5
            reconstructed_obs = torch.clamp(reconstructed_obs / 255.0 - 0.5, -0.5, 0.5)
            reconstruction_loss = self.reconstruction_loss_fn(reconstructed_obs, obs)
            reconstruction_loss = (
                reconstruction_loss.reshape(obs.shape[0], -1).sum(dim=1).mean()
            )

            loss = critic_loss + reconstruction_loss * self.reconstruction_loss_coeff

            self.encoder_opt.zero_grad(set_to_none=True)
            self.decoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()
            self.decoder_opt.step()
        elif self.use_decoder:
            # Detach encoder_obs so we don't backprop reconstruction loss through encoder
            reconstructed_obs = self.decoder(encoded_obs.detach())
            obs = obs / 255.0 - 0.5
            reconstructed_obs = torch.clamp(reconstructed_obs / 255.0 - 0.5, -0.5, 0.5)
            reconstruction_loss = self.reconstruction_loss_fn(reconstructed_obs, obs)
            reconstruction_loss = (
                reconstruction_loss.reshape(obs.shape[0], -1).sum(dim=1).mean()
            )

            self.decoder_opt.zero_grad(set_to_none=True)
            reconstruction_loss.backward()
            self.decoder_opt.step()

            self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()
        else:
            self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        if (
            self.disentangled_version == 1 or self.disentangled_version == 3
        ) and self.use_tb:
            metrics["reconstruction_loss"] = reconstruction_loss.item()
            metrics["mask_loss"] = mask_loss.item()
        elif self.disentangled_version == 2 and self.use_tb:
            metrics["non_robot_mask_loss"] = non_robot_mask_loss.item()
            metrics["robot_mask_loss"] = robot_mask_loss.item()
        elif self.use_decoder and self.use_tb:
            metrics["reconstruction_loss"] = reconstruction_loss.item()

        return metrics

    def update_actor(self, encoded_obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(encoded_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(encoded_obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = batch
        rgb_obs = obs["pixels"]
        next_rgb_obs = next_obs["pixels"]
        rgb_obs, action, reward, discount, next_rgb_obs = utils.to_torch(
            (rgb_obs, action, reward, discount, next_rgb_obs), self.device
        )

        if self.disentangled_version > 0:
            masks = obs["segmentation"]
            (masks,) = utils.to_torch((masks,), self.device)
        else:
            masks = None

        # augment
        rgb_obs = self.aug(rgb_obs.float())
        next_rgb_obs = self.aug(next_rgb_obs.float())
        if masks is not None:
            masks = self.aug(masks.float())
        # encode
        encoded_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_rgb_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic and decoder
        metrics.update(
            self.update_critic_and_decoder(
                rgb_obs,
                encoded_obs,
                action,
                reward,
                discount,
                encoded_next_obs,
                step,
                masks,
            )
        )

        # update actor
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
