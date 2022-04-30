import pandas as pd
import tqdm
import random
import bird_view.utils.carla_utils as cu
from fuzz.fuzz import fuzzing
import numpy as np
import copy
import time
import pickle

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    reward = 0.0
    reward += np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    reward += 0.2 * (cur_speed_norm - prev_speed_norm)
    # print('speed: ', cur_speed_norm)
    if cur_collid:
        reward -= 100 * cur_speed_norm
    if cur_invade:
        reward -= cur_speed_norm

    return reward

def run_single(env, weather, start, target, agent_maker, seed):
    # HACK: deterministic vehicle spawns.
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    env.replay = True
    # HACK: finish here
    env.replayer = pickle.load()

    while len(env.replayer.corpus) > 0:
        env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
        start_pose = env._start_pose

        diagnostics = list()
        result = {
            "weather": weather,
            "start": start,
            "target": target,
            "success": None,
            "t": None,
            "total_lights_ran": None,
            "total_lights": None,
            "collided": None,
        }
        seq_entropy = 0

        first_reward_flag = True
        total_reward = 0
        sequence = []
        while env.tick():
            observations = env.get_observations()
            if first_reward_flag == False:
                cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                cur_collid = (prev_collided_frame_number != env._collided_frame_number)
            if first_reward_flag:
                first_reward_flag = False
                prev_distance = env._local_planner.distance_to_goal
                prev_speed = observations['velocity']
                prev_invaded_frame_number = env._invaded_frame_number
                prev_collided_frame_number = env._collided_frame_number
                cur_invade = False
                cur_collid = False
                if env.invaded:
                    cur_invade = True
                if env.collided:
                    cur_collid = True

            reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
            total_reward += reward
            prev_distance = env._local_planner.distance_to_goal
            prev_speed = observations['velocity']
            prev_invaded_frame_number = env._invaded_frame_number
            prev_collided_frame_number = env._collided_frame_number

            control, current_entropy = agent.run_step(observations)

            seq_entropy += current_entropy
            diagnostic = env.apply_control(control)
            diagnostic.pop("viz_img")
            diagnostics.append(diagnostic)

            if env.is_failure() or env.is_success() or env._tick > 100:
                result["success"] = env.is_success()
                result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                result["total_lights"] = env.traffic_tracker.total_lights
                result["collided"] = env.collided
                result["t"] = env._tick
                break
        print(total_reward)

    return result, diagnostics

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as handle:
        replayer = pickle.load(handle)
    corpus = []
    total_crash = len(replayer.corpus)
    print(replayer.corpus[0])

def replay_stored(agent_maker, env, benchmark_dir, seed, resume, max_run=5):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / "summary.csv"
    diagnostics_dir = benchmark_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    for weather, (start, target), run_name in env.all_tasks:
        if (
            resume
            and len(summary) > 0
            and (
                (summary["start"] == start)
                & (summary["target"] == target)
                & (summary["weather"] == weather)
            ).any()
        ):
            print(weather, start, target)
            continue

        diagnostics_csv = str(diagnostics_dir / ("%s.csv" % run_name))

        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed)

        summary = summary.append(result, ignore_index=True)

        # Do this every timestep just in case.
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)

        break

        # num_run += 1

        # if num_run >= max_run:
        #     break

