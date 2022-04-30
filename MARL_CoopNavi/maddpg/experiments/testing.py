if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import copy
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTester
import tensorflow.contrib.layers as layers
from fuzz.fuzz import fuzzing
import tqdm, sys

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=300000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='spread', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False, fuzz=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if fuzz:
        env = MultiAgentEnv(world, scenario.reset_world_fuzz, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done_flag, verify_func=scenario.verify)
    else:
        env = MultiAgentEnv(world, scenario.reset_world_before_fuzz, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done_flag)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTester
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def get_observe(env):
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
        state.append(agent.state.p_vel)
        state.append(agent.state.c)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return list(np.array(state).flatten())

def get_init_state(env):
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return state

def get_collision_num(env):
    collisions = 0
    for i, agent in enumerate(env.world.agents):
        for j, agent_other in enumerate(env.world.agents):
            if i == j:
                continue
            delta_pos = agent.state.p_pos - agent_other.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = (agent.size + agent_other.size)
            if dist < dist_min:
                collisions += 1
    return collisions / 2

def sample_seeds(seed_num=3000):
    fuzzer = fuzzing()
    np.random.seed(2021)
    pbar = tqdm.tqdm(total=seed_num)
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark, fuzz=False)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        U.initialize()
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        U.load_state(arglist.load_dir)
        episode_rewards = 0  # sum of rewards for all agents
        obs_n = env.reset()
        episode_step = 0
        seed_index = 0
        state_seqs = []
        state_seqs.append(get_observe(env))
        collisions = 0
        init_state = get_init_state(env)
        while seed_index < seed_num:
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            state_seqs.append(get_observe(env))
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            obs_n = new_obs_n
            collisions += get_collision_num(env)

            for i, rew in enumerate(rew_n):
                episode_rewards += rew

            if done or terminal:
                cvg = fuzzer.state_coverage(state_seqs)
                fuzzer.further_mutation(copy.deepcopy(init_state), episode_rewards, 1.0, cvg, copy.deepcopy(init_state))
                print(episode_rewards, terminal, done, collisions, len(fuzzer.corpus))
                obs_n = env.reset()
                state_seqs = []
                init_state = get_init_state(env)
                state_seqs.append(get_observe(env))
                seed_index += 1
                episode_step = 0
                collisions = 0
                episode_rewards = 0
                pbar.update(1)

        env_uncertainty = make_env(arglist.scenario, arglist, arglist.benchmark, fuzz=True)
        for i in range(len(fuzzer.corpus)):
            current_frame = fuzzer.corpus[i]
            obs_n = env_uncertainty.reset(current_frame[0:3], current_frame[3:])
            mutated_frame = fuzzer.mutate(copy.deepcopy(current_frame))
            episode_step = 0
            uncertainity_reward = 0
            done = False
            terminal = False
            while True:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, info_n = env_uncertainty.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                obs_n = new_obs_n
                for rew in rew_n:
                    uncertainity_reward += rew
                if done or terminal:
                    fuzzer.entropy[i] = np.abs(fuzzer.rewards[i] - uncertainity_reward) / 100
                    break

    return fuzzer, trainers

def test(arglist):
    fuzzer, trainers = sample_seeds(3000)
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark, fuzz=True)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        U.initialize()
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        U.load_state(arglist.load_dir)

        episode_rewards = 0.0  # sum of rewards for all agents
        start_fuzz_time = time.time()
        current_fuzz_time = time.time()

        # while current_fuzz_time - start_fuzz_time < 3600 * 12:
        current_pos = fuzzer.get_pose()
        new_pos = fuzzer.mutate(current_pos)
        obs_n = env.reset(new_pos[0:3], new_pos[3:])
        agent_flag, landmark_flag = env.verify_func(env.world)
        while agent_flag or landmark_flag:
            new_pos = fuzzer.mutate(current_pos)
            obs_n = env.reset(new_pos[0:3], new_pos[3:])
            agent_flag, landmark_flag = env.verify_func(env.world)

        state_seqs = []
        state_seqs.append(get_observe(env))
        collisions = 0
        init_state = get_init_state(env)
        episode_step = 0
        train_step = 0

        # HACK: set time here
        while current_fuzz_time - start_fuzz_time < 3600 * 1 and len(fuzzer.corpus) > 0:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            state_seqs.append(get_observe(env))
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            collisions += get_collision_num(env)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards += rew
            if terminal and collisions > 5 and not done:
                print('Found: ', len(fuzzer.result))
                fuzzer.add_crash(copy.deepcopy(init_state))
            elif done or terminal:
                cvg = fuzzer.state_coverage(state_seqs)
                current_fuzz_time = time.time()
                print('Time: ', current_fuzz_time - start_fuzz_time, ' reward: ', episode_rewards, ' cvg: ', cvg)
                if episode_rewards < fuzzer.current_reward or cvg < 0.01:
                    fuzzer.further_mutation(copy.deepcopy(init_state), episode_rewards, np.abs(episode_rewards - fuzzer.current_reward) / 100.0, cvg, fuzzer.current_original)
            if done or terminal:
                current_fuzz_time = time.time()
                current_pos = fuzzer.get_pose()
                new_pos = fuzzer.mutate(current_pos)
                obs_n = env.reset(new_pos[0:3], new_pos[3:])
                agent_flag, landmark_flag = env.verify_func(env.world)
                mutation_count = 0
                while agent_flag or landmark_flag:
                    mutation_count += 1
                    new_pos = fuzzer.mutate(current_pos)
                    obs_n = env.reset(new_pos[0:3], new_pos[3:])
                    agent_flag, landmark_flag = env.verify_func(env.world)
                    if mutation_count > 10 and (agent_flag or landmark_flag):
                        fuzzer.drop_current()
                        mutation_count = 0
                        current_pos = fuzzer.get_pose()
                state_seqs = []
                init_state = get_init_state(env)
                state_seqs.append(get_observe(env))
                episode_step = 0
                collisions = 0
                episode_rewards = 0
    
    with open('./results/crash_artifact_fuzzing.pkl', 'wb') as handle:
        pickle.dump(fuzzer.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    np.random.seed(2021)
    arglist = parse_args()
    f = open('./results/artifact_fuzzing.txt', 'w', buffering=1)
    sys.stdout = f
    test(arglist)
