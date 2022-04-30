import numpy as np
import torch
from collections import deque, namedtuple
import cv2
import os
import carla

from .model_supervised import Model_Segmentation_Traffic_Light_Supervised
from .model_RL import DQN, Orders


class AgentIAsRL:
    def __init__(self, args=None, **kwargs):
        super().__init__(**kwargs)

        self.args = args

        path_to_folder_with_model = args.path_folder_model
        path_to_model_supervised = os.path.join(path_to_folder_with_model, "model_supervised/")
        path_model_supervised = None
        for file in os.listdir(path_to_model_supervised):
            if ".pth" in file:
                if path_model_supervised is not None:
                    raise ValueError(
                        "There is multiple model supervised in folder " +
                        path_to_model_supervised +
                        " you must keep only one!",
                    )
                path_model_supervised = os.path.join(path_to_model_supervised, file)
        if path_model_supervised is None:
            raise ValueError("We didn't find any model supervised in folder " +
                             path_to_model_supervised)

        # All this magic number should match the one used when training supervised...
        model_supervised = Model_Segmentation_Traffic_Light_Supervised(
            len(args.steps_image), len(args.steps_image), 1024, 6, 4, args.crop_sky
        )
        model_supervised.load_state_dict(
            torch.load(path_model_supervised, map_location=args.device)
        )
        model_supervised.to(device=args.device)

        self.encoder = model_supervised.encoder
        self.last_conv_downsample = model_supervised.last_conv_downsample

        self.action_space = (args.nb_action_throttle + 1) * args.nb_action_steering

        path_to_model_RL = os.path.join(path_to_folder_with_model, "model_RL")
        os.chdir(path_to_model_RL)
        tab_model = []
        for file in os.listdir(path_to_model_RL):
            if ".pth" in file:
                tab_model.append(os.path.join(path_to_model_RL, file))

        if len(tab_model) == 0:
            raise ValueError("We didn't find any RL model in folder "+ path_to_model_RL)

        self.tab_RL_model = []
        for current_model in tab_model:

            current_RL_model = DQN(args, self.action_space).to(device=args.device)
            current_RL_model_dict = current_RL_model.state_dict()

            # print("we load RL model ", current_model)
            checkpoint = torch.load(current_model)

            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in checkpoint["model_state_dict"].items()
                if k in current_RL_model_dict
            }
            # 2. overwrite entries in the existing state dict
            current_RL_model_dict.update(pretrained_dict)
            # 3. load the new state dict
            current_RL_model.load_state_dict(current_RL_model_dict)
            self.tab_RL_model.append(current_RL_model)

        self.window = (
            max([abs(number) for number in args.steps_image]) + 1
        )  # Number of frames to concatenate
        self.RGB_image_buffer = deque([], maxlen=self.window)
        self.device = args.device

        self.state_buffer = deque([], maxlen=self.window)
        self.State = namedtuple("State", ("image", "speed", "order", "steering"))

        if args.crop_sky:
            blank_state = self.State(
                np.zeros(6144, dtype=np.float32), -1, -1, 0
            )  # RGB Image, color channet first for torch
        else:
            blank_state = self.State(np.zeros(8192, dtype=np.float32), -1, -1, 0)
        for _ in range(self.window):
            self.state_buffer.append(blank_state)
            if args.crop_sky:
                self.RGB_image_buffer.append(
                    np.zeros((3, args.front_camera_height - 120, args.front_camera_width))
                )
            else:
                self.RGB_image_buffer.append(
                    np.zeros((3, args.front_camera_height, args.front_camera_width))
                )

        self.last_steering = 0
        self.last_order = 0

        self.current_timestep = 0

    def act(self, state_buffer, RL_model):
        speeds = []
        order = state_buffer[-1].order
        steerings = []
        for step_image in self.args.steps_image:
            state = state_buffer[step_image + self.window - 1]
            speeds.append(state.speed)
            steerings.append(state.steering)
        images = torch.from_numpy(state_buffer[-1].image).to(self.device, dtype=torch.float32)
        speeds = torch.from_numpy(np.stack(speeds).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        steerings = torch.from_numpy(np.stack(steerings).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        with torch.no_grad():
            quantile_values, _, temps = RL_model(
                images.unsqueeze(0),
                speeds.unsqueeze(0),
                order,
                steerings.unsqueeze(0),
                self.args.num_quantile_samples,
            )

            # HACK: store temps and corresponding labels
            temps = temps.cpu().detach().numpy()

            return quantile_values.mean(0).argmax(0).item(), temps

    # We had different mapping int/order in our training than in the CARLA benchmark,
    # so we need to remap orders
    def adapt_order(self, incoming_obs_command):
        if incoming_obs_command == 1:  # LEFT
            return Orders.Left.value
        if incoming_obs_command == 2:  # RIGHT
            return Orders.Right.value
        if incoming_obs_command == 3:  # STRAIGHT
            return Orders.Straight.value
        if incoming_obs_command == 4:  # FOLLOW_LANE
            return Orders.Follow_Lane.value

    def run_step(self, observations):
        self.current_timestep += 1
        rgb = observations["rgb"].copy()
        if self.args.crop_sky:
            rgb = np.array(rgb)[120:, :, :]
        else:
            rgb = np.array(rgb)
        if self.args.render:
            bgr = rgb[:, :, ::-1]
            # bgr = observations["road"]
            cv2.imshow("network input", bgr)
            cv2.waitKey(1)

        rgb = np.rollaxis(rgb, 2, 0)
        self.RGB_image_buffer.append(rgb)

        speed = np.linalg.norm(observations["velocity"])

        order = self.adapt_order(int(observations["command"]))
        if self.last_order != order:
            # print("order = ", Orders(order).name)
            self.last_order = order

        np_array_RGB_input = np.concatenate(
            [
                self.RGB_image_buffer[indice_image + self.window - 1]
                for indice_image in self.args.steps_image
            ]
        )
        torch_tensor_input = (
            torch.from_numpy(np_array_RGB_input)
            .to(dtype=torch.float32, device=self.device)
            .div_(255)
            .unsqueeze(0)
        )

        with torch.no_grad():
            current_encoding = self.encoder(torch_tensor_input)

            current_encoding = self.last_conv_downsample(current_encoding)

        current_encoding_np = current_encoding.cpu().numpy().flatten()

        current_state = self.State(current_encoding_np, speed, order, self.last_steering)
        self.state_buffer.append(current_state)

        tab_action = []

        current_temps = []
        for RL_model in self.tab_RL_model:
            current_action, temps = self.act(self.state_buffer, RL_model)
            tab_action.append(current_action)
            current_temps.append(temps)

        steer = 0
        throttle = 0
        brake = 0

        for action in tab_action:

            steer += (
                (action % self.args.nb_action_steering) - int(self.args.nb_action_steering / 2)
            ) * (self.args.max_steering / int(self.args.nb_action_steering / 2))
            if action < int(self.args.nb_action_steering * self.args.nb_action_throttle):
                throttle += (int(action / self.args.nb_action_steering)) * (
                    self.args.max_throttle / (self.args.nb_action_throttle - 1)
                )
                brake += 0
            else:
                throttle += 0
                brake += 1.0

        steer = steer / len(tab_action)
        throttle = throttle / len(tab_action)
        if brake < len(tab_action) / 2:
            brake = 0
        else:
            brake = brake / len(tab_action)
        
        # 8 models here
        # HACK: local sensitivity
        current_entropy = np.std(np.array(tab_action))

        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False
        self.last_steering = steer
        return control, current_entropy, current_temps
