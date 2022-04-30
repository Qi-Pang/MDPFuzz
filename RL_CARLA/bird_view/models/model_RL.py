import math
import torch
from torch import nn
from torch.nn import functional as F


# ORDER
from enum import Enum

Orders = Enum("Order", "Follow_Lane Straight Right Left")


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, disable_cuda=False):
        super(NoisyLinear, self).__init__()
        self.disable_cuda = disable_cuda
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        if self.disable_cuda:
            x = torch.randn(size)
        else:
            x = torch.cuda.FloatTensor(size).normal_()
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super().__init__()
        self.action_space = action_space
        self.history_length = len(args.steps_image)
        self.device = args.device
        self.disable_cuda = args.disable_cuda

        self.magic_number_repeat_scaler_in_fc = 10

        self.magic_number_SCALE_steering_in_fc = 10  # We want to multiply by 10 the steering...

        self.quantile_embedding_dim = args.quantile_embedding_dim

        if args.crop_sky:
            size_RL_state = 6144
        else:
            size_RL_state = 8192
        self.iqn_fc = nn.Linear(self.quantile_embedding_dim, size_RL_state)

        hidden_size = 1024

        self.fcnoisy_h_a = NoisyLinear(size_RL_state, hidden_size)

        hidden_size2 = 512

        self.fcnoisy0_z_a_lane_follow = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length,
            hidden_size2,
        )
        self.fcnoisy0_z_a_straight = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length,
            hidden_size2,
        )
        self.fcnoisy0_z_a_right = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length,
            hidden_size2,
        )
        self.fcnoisy0_z_a_left = NoisyLinear(
            hidden_size + 2 * self.magic_number_repeat_scaler_in_fc * self.history_length,
            hidden_size2,
        )

        self.fcnoisy1_z_a_lane_follow = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_straight = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_right = NoisyLinear(hidden_size2, action_space)
        self.fcnoisy1_z_a_left = NoisyLinear(hidden_size2, action_space)

    def forward(self, images, speeds, orders, steerings, num_quantiles):

        batch_size = images.shape[0]

        quantiles = torch.cuda.FloatTensor(num_quantiles * batch_size, 1).uniform_(0, 1)

        quantile_net = quantiles.repeat([1, self.quantile_embedding_dim])

        quantile_net = torch.cos(
            torch.arange(
                1,
                self.quantile_embedding_dim + 1,
                1,
                device=torch.device("cuda"),
                dtype=torch.float32,
            )
            * math.pi
            * quantile_net
        )

        quantile_net = self.iqn_fc(quantile_net)
        quantile_net = F.relu(quantile_net)

        rl_state_net = images.repeat(num_quantiles, 1)
        rl_state_net = rl_state_net * quantile_net

        mask_lane_follow = orders == Orders.Follow_Lane.value
        mask_straight = orders == Orders.Straight.value
        mask_right = orders == Orders.Right.value
        mask_left = orders == Orders.Left.value

        if batch_size != 1:
            mask_lane_follow = mask_lane_follow.float()[:, None].repeat(num_quantiles, 1)
            mask_straight = mask_straight.float()[:, None].repeat(num_quantiles, 1)
            mask_right = mask_right.float()[:, None].repeat(num_quantiles, 1)
            mask_left = mask_left.float()[:, None].repeat(num_quantiles, 1)

        else:
            mask_lane_follow = bool(mask_lane_follow)
            mask_straight = bool(mask_straight)
            mask_right = bool(mask_right)
            mask_left = bool(mask_left)

        just_before_order_heads_a = F.relu(self.fcnoisy_h_a(rl_state_net))

        steerings = steerings * self.magic_number_SCALE_steering_in_fc

        speeds = speeds.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)
        steerings = steerings.repeat(num_quantiles, self.magic_number_repeat_scaler_in_fc)

        just_before_order_heads_a_plus_speed_steering = torch.cat(
            (just_before_order_heads_a, speeds, steerings), 1
        )

        a_lane_follow = self.fcnoisy0_z_a_lane_follow(
            just_before_order_heads_a_plus_speed_steering
        )
        temp_1 = torch.sign(F.relu(a_lane_follow).clone())
        temp_1 = torch.mean(temp_1, axis=0)
        a_lane_follow = self.fcnoisy1_z_a_lane_follow(F.relu(a_lane_follow))

        a_straight = self.fcnoisy0_z_a_straight(just_before_order_heads_a_plus_speed_steering)
        temp_2 = torch.sign(F.relu(a_straight).clone())
        temp_2 = torch.mean(temp_2, axis=0)
        a_straight = self.fcnoisy1_z_a_straight(F.relu(a_straight))

        a_right = self.fcnoisy0_z_a_right(just_before_order_heads_a_plus_speed_steering)
        temp_3 = torch.sign(F.relu(a_right).clone())
        temp_3 = torch.mean(temp_3, axis=0)
        a_right = self.fcnoisy1_z_a_right(F.relu(a_right))

        a_left = self.fcnoisy0_z_a_left(just_before_order_heads_a_plus_speed_steering)
        temp_4 = torch.sign(F.relu(a_left).clone())
        temp_4 = torch.mean(temp_4, axis=0)
        a_left = self.fcnoisy1_z_a_left(F.relu(a_left))

        a = (
            a_lane_follow * mask_lane_follow
            + a_straight * mask_straight
            + a_right * mask_right
            + a_left * mask_left
        )

        return a, quantiles, temp_1 * mask_lane_follow + temp_2 * mask_straight + temp_3 * mask_right + temp_4 * mask_left

    def reset_noise(self):
        for name, module in self.named_children():
            if "fcnoisy" in name:
                module.reset_noise()
