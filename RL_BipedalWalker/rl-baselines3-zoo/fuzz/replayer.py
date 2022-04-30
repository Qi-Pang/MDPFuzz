import numpy as np
from scipy.stats import multivariate_normal
import copy
import tqdm
import carla

class replayer:
    def __init__(self):
        self.corpus = []
        self.rewards = []
        # self.result = []
        self.entropy = []
        self.coverage = []
        self.original = []
        # self.count = []
        self.envsetting = []
        self.state_cvg = []

        self.sequences = []
        self.current_pose = None
        self.current_reward = None
        self.current_entropy = None
        self.current_coverage = None
        # self.current_original = None
        self.current_index = None
        self.current_nvsetting = None
        self.replay_list = None

    def get_pose(self):
        # choose_index = np.random.choice(range(len(self.corpus)), 1)[0]
        if self.replay_list == None:
            choose_index = 0
        else:
            choose_index = self.replay_list[-1]
            self.replay_list.pop(len(self.replay_list)-1)

        self.current_index = choose_index
        self.current_pose = self.corpus[choose_index][0]
        self.current_vehicle_info = self.corpus[choose_index][1]
        self.current_reward = self.rewards[choose_index]
        self.current_entropy = self.entropy[choose_index]
        self.current_coverage = self.coverage[choose_index]
        # self.current_original = self.original[choose_index]
        self.current_envsetting = self.envsetting[choose_index]

        self.corpus.pop(choose_index)
        self.rewards.pop(choose_index)
        self.entropy.pop(choose_index)
        self.coverage.pop(choose_index)
        # self.original.pop(choose_index)
        # self.count.pop(choose_index)
        self.envsetting.pop(choose_index)
        self.current_index = None

        return self.current_pose

    def get_vehicle_info(self):
        return self.current_vehicle_info

    def store(self, current_pose, rewards, entropy, cvg, original, further_envsetting):
        pose = current_pose[0]
        newpose = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
        vehicle_info = current_pose[1]
        new_vehicle_info = []
        for i in range(len(vehicle_info)):
            pose = vehicle_info[i][1]
            v_1 = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
            temp = (vehicle_info[i][0], v_1, vehicle_info[i][2], vehicle_info[i][3])
            new_vehicle_info.append(temp)

        self.corpus.append((newpose, new_vehicle_info))
        self.rewards.append(rewards)
        self.entropy.append(entropy)
        self.coverage.append(cvg)
        self.original.append(original)
        self.envsetting.append(further_envsetting)

    def drop_current(self):
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.coverage.pop(choose_index)
            # self.original.pop(choose_index)
            # self.count.pop(choose_index)
            self.envsetting.pop(choose_index)
            self.current_index = None
