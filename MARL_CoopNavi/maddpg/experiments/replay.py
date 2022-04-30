import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTester
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
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

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world_fuzz, scenario.reward, scenario.observation, scenario.benchmark_data, scenario.done_flag, verify_func=scenario.verify)
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

def replay(arglist, pickle_path):
    with open(pickle_path, 'rb') as handle:
        result = pickle.load(handle)
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        U.load_state(arglist.load_dir)

        episode_rewards = []  # sum of rewards for all agents

        for j in range(len(result)):
            obs_n = env.reset(result[j][0:3], result[j][3:])
            episode_step = 0
            train_step = 0
            collisions = get_collision_num(env)
            episode_rewards.append(0)
            t_start = time.time()

            print('Starting iterations...')
            while True:
                # get action
                action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                collisions += get_collision_num(env)
                terminal = (episode_step >= arglist.max_episode_len)
                obs_n = new_obs_n
                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew

                if done or terminal:
                    print(j, episode_rewards[-1], collisions)
                    break

                # for displaying learned policies
                if arglist.display:
                    time.sleep(0.1)
                    env.render()
                    continue

if __name__ == '__main__':
    np.random.seed(2021)
    arglist = parse_args()
    pickle_path = './results/crash.pkl'
    replay(arglist, pickle_path)
