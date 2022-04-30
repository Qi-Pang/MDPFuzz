from torch.utils import data
from torchvision import transforms as T
import numpy as np
import pickle, tqdm
from simulate import ACASagent, Autoagent, env, calculate_init_bounds
import copy, sys

class NocrashEnv:
    def __init__(self, acas_speed, x2, y2, auto_theta, auto_speed):
        self.ownship = ACASagent(acas_speed)
        self.inturder = Autoagent(x2, y2, auto_theta, auto_speed)
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed

    def update_params(self):
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed
        self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row)
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta

        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2

    def step(self):
        acas_act = self.ownship.act([self.row, self.alpha, self.phi, self.Vown, self.Vint])
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()

    def step_proof(self, direction):
        acas_act = self.ownship.act_proof(direction)
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()

def reward_func(acas_speed, x2, y2, auto_theta, auto_speed):
    dis_threshold = 200
    air1 = NocrashEnv(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta, auto_speed=auto_speed)
    air1.update_params()
    gamma = 0.99

    min_dis1 = np.inf
    reward = 0

    collide_flag = False
    states_seq = []

    for j in range(100):
        air1.step()
        reward = reward * gamma + air1.row / 60261.0
        # states_seq.append(normalize_state([air1.row, air1.alpha, air1.phi, air1.Vown, air1.Vint]))
        if air1.row < dis_threshold:
            collide_flag = True
            reward -= 100

    return reward, collide_flag, states_seq


class ACAS_data(data.Dataset):
    def __init__(self, crash_data, nocrash_data, mode='train'):
        np.random.seed(2021)
        self.data = np.vstack((crash_data, nocrash_data))
        np.random.shuffle(self.data)
        self.transforms = T.Compose([
            T.ToTensor(),
        ])
        self.mode = mode
        spilt_point = int(0.8 * self.data.shape[0])
        # self.train_data = self.data[:spilt_point]
        # self.train_data = copy.deepcopy(self.data)
        self.train_data = copy.deepcopy(crash_data)
        # self.test_data = self.data[spilt_point:]
        self.test_data = crash_data

    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.train_data[index][:5]
            data = data.reshape(1, data.shape[0]).astype(np.float32)
            data = self.transforms(data)
            label = int(self.train_data[index][6])
        else:
            data = self.test_data[index][:5]
            data = data.reshape(1, data.shape[0]).astype(np.float32)
            data = self.transforms(data)
            label = int(self.test_data[index][6])
        return data, label

    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

def sample_nocrash(num_seeds):
    np.random.seed(2021)
    dis_threshold = 200
    pbar = tqdm.tqdm(total=num_seeds)
    nocrash_data = []
    # active = []

    while len(nocrash_data) < num_seeds:
        temp = []
        # temp_active = []
        acas_speed = np.random.uniform(low=10, high=1100)
        row = np.random.uniform(low=1000, high=60261)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x2 = row * np.cos(theta)
        y2 = row * np.sin(theta)
        bound1, bound2 = calculate_init_bounds(x2, y2)
        auto_theta = np.random.uniform(low=bound1, high=bound2)
        auto_speed = np.random.uniform(low=0, high=1200)

        air_nocrash = NocrashEnv(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta, auto_speed=auto_speed)
        air_nocrash.update_params()
        min_dis = air_nocrash.row
        for j in range(100):
            if j > 0:
                temp.append([air_nocrash.row, air_nocrash.alpha, air_nocrash.phi, air_nocrash.Vown, air_nocrash.Vint, air_nocrash.ownship.prev_action, 0])
                # temp_active.append(copy.deepcopy(air_nocrash.ownship.current_active))
            air_nocrash.step()
            if j > 0:
                temp[j-1][6] = air_nocrash.ownship.prev_action
            if min_dis > air_nocrash.row:
                min_dis = air_nocrash.row
        if min_dis > dis_threshold:
            # reward, _, _ = reward_func(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta, auto_speed=auto_speed)
            nocrash_data.append(copy.deepcopy(temp))
            # active.append(temp_active)
            pbar.update(1)
    return np.array(nocrash_data).reshape(-1, 7)

def direction(acas_speed, x2, y2, auto_theta):
    dis_threshold = 200
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()

    air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air2.update_params()

    min_dis1 = air1.row
    min_dis2 = air2.row
    for j in range(100):
        air1.step_proof(3)
        if min_dis1 > air1.row:
            min_dis1 = air1.row
    for j in range(100):
        air2.step_proof(4)
        if min_dis2 > air2.row:
            min_dis2 = air2.row
    if min_dis1 > min_dis2:
        return 3
    else:
        return 4

def generate_data(pickle_path):
    with open(pickle_path, 'rb') as handle:
        result = pickle.load(handle)
    crash_data = []
    count_3 = 0
    count_4 = 0
    # crash_active = []
    for i in range(len(result)):
        [acas_speed, x2, y2, auto_theta] = result[i]
        label = direction(acas_speed, x2, y2, auto_theta)
        if label == 3:
            count_3 += 1
        if label == 4:
            count_4 += 1
        air = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air.update_params()
        for j in range(100):
            crash_data.append([air.row, air.alpha, air.phi, air.Vown, air.Vint, air.ownship.prev_action, label])
            air.step()
            # crash_active.append(copy.deepcopy(air.ownship.current_active))
    crash_data = np.array(crash_data)
    # return crash_data, crash_active
    return crash_data

def generate_target_data(pickle_path):
    with open(pickle_path, 'rb') as handle:
        result = pickle.load(handle)
    crash_data_3 = []
    crash_data_4 = []
    count_3 = 0
    count_4 = 0
    for i in range(len(result)):
        [acas_speed, x2, y2, auto_theta] = result[i]
        label = direction(acas_speed, x2, y2, auto_theta)
        if label == 3:
            count_3 += 1
        if label == 4:
            count_4 += 1
        air = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air.update_params()
        if label == 3:
            for j in range(100):
                crash_data_3.append([air.row, air.alpha, air.phi, air.Vown, air.Vint, air.ownship.prev_action, label])
                air.step()
        else:
            for j in range(100):
                crash_data_4.append([air.row, air.alpha, air.phi, air.Vown, air.Vint, air.ownship.prev_action, label])
                air.step()
    crash_data_3 = np.array(crash_data_3)
    crash_data_4 = np.array(crash_data_4)
    return crash_data_3, crash_data_4

def sample_random(num_seeds):
    np.random.seed(2021)
    dis_threshold = 200
    pbar = tqdm.tqdm(total=num_seeds)
    nocrash_data = []
    active = []

    while len(nocrash_data) < num_seeds:
        temp = []
        temp_active = []
        acas_speed = np.random.uniform(low=10, high=1100)
        row = np.random.uniform(low=1000, high=60261)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x2 = row * np.cos(theta)
        y2 = row * np.sin(theta)
        bound1, bound2 = calculate_init_bounds(x2, y2)
        auto_theta = np.random.uniform(low=bound1, high=bound2)
        # auto_speed = np.random.uniform(low=0, high=1200)

        air_random = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air_random.update_params()
        min_dis = air_random.row
        for j in range(100):
            if j > 0:
                temp.append([air_random.row, air_random.alpha, air_random.phi, air_random.Vown, air_random.Vint, air_random.ownship.prev_action, 0])
                temp_active.append(copy.deepcopy(air_random.ownship.current_active))
            air_random.step()
            if j > 0:
                temp[j-1][6] = air_random.ownship.prev_action
            if min_dis > air_random.row:
                min_dis = air_random.row
        # reward, _, _ = reward_func(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta, auto_speed=auto_speed)
        nocrash_data.append(copy.deepcopy(temp))
        active.append(temp_active)
        pbar.update(1)
    return np.array(nocrash_data).reshape(-1, 7), active



if __name__ == '__main__':
    '''
    crash_data = generate_data('./results/crash.pkl')
    print(crash_data.shape)
    nocrash_data = sample_nocrash(200)
    print(nocrash_data.shape)
    training_set = ACAS_data(crash_data, nocrash_data, 'train')
    # print(training_set.data[15])
    test_set = ACAS_data(crash_data, nocrash_data, 'test')
    # print(test_set.data[15])
    '''
    # f = open('./results/reward_orig_log.txt', 'w', buffering=1)
    # sys.stdout = f
    # _, active = sample_nocrash(136)
    # _, crash_active = generate_data('./results/crash_0712.pkl')
    _, crash_active = sample_random(136)
    # with open('./results/tsne_nocrash.pkl', 'wb') as handle:
    #     pickle.dump(active, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./results/tsne_random.pkl', 'wb') as handle:
        pickle.dump(crash_active, handle, protocol=pickle.HIGHEST_PROTOCOL)
