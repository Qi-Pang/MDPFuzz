import pandas as pd
import tqdm
import random
import bird_view.utils.carla_utils as cu
from fuzz.fuzz import fuzzing
from fuzz.replayer import replayer
import numpy as np
import copy
import time
import pickle

import carla
from carla import ColorConverter
from carla import WeatherParameters

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    reward = 0.0
    reward += np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    reward += 0.2 * (cur_speed_norm - prev_speed_norm)
    if cur_collid:
        reward -= 100 * cur_speed_norm
    if cur_invade:
        reward -= cur_speed_norm

    return reward

def collect_corpus():
    return None

def get_index(sequence):
    result = np.zeros(17)
    mins = [-10, 90, -1, -1, -10, -10, -1, -20, -20, -5, -10, 90, 0, 1, 50, 50, 0]
    maxs = [200, 350, 1, 1, 10, 10, 1, 20, 20, 5, 200, 350, 0.5, 4, 300, 300, 15]
    for i in range(sequence.shape[0]):
        if sequence[i] < mins[i]:
            result[i] = 0
        elif sequence[i] > maxs[i]:
            result[i] = 10000
        else:
            result_i = (sequence[i] - mins[i]) / (maxs[i] - mins[i]) * 10000 + 1
            if np.isnan(result_i):
                result[i] = 10000
            else:
                result[i] = int(result_i)
    return result

def update_dict(storage, sequences):
    for i in range(len(sequences)):
        index = get_index(sequences[i])
        for j in range(index.shape[0]):
            storage[int(j * 10000 + index[j])] = 1
    return storage

def run_single(env, weather, start, target, agent_maker, seed, replay=False, em_guide=True):
    # HACK: deterministic vehicle spawns.
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    print('replay = ', replay, 'em guide = ', em_guide)
    if replay == True:
        prints = env._blueprints.filter("vehicle.*")
        replayer_load = load_pickle('./results/EM_crash.pkl', prints)
        print('Load Done!')
        env.replay = True
        env.replayer = replayer_load
        print(env.replayer.rewards)
        env.replayer.replay_list = [14, 24, 26, 28, 29, 30, 32, 34, 41, 42, 44, 45, 46, 54, 64, 66, 70, 71, 73, 75, 76, 78, 80, 81, 83, 95]
        # env.replayer.replay_list = [73]
        agent = agent_maker()
        tsne_data = []

        while len(env.replayer.replay_list) > 0:
            print('current id: ', env.replayer.replay_list[-1])
            env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
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

                control, current_entropy, _ = agent.run_step(observations)
                seq_entropy += current_entropy
                diagnostic = env.apply_control(control)
                diagnostic.pop("viz_img")
                diagnostics.append(diagnostic)
                # HACK: T-SNE
                # current_tsne = np.array(current_tsne).flatten()
                # tsne_data.append(current_tsne)
                if env.is_failure() or env.is_success() or env._tick > 100:
                    result["success"] = env.is_success()
                    result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                    result["total_lights"] = env.traffic_tracker.total_lights
                    result["collided"] = env.collided
                    result["t"] = env._tick
                    break
            print(total_reward)
        
        # HACK: T-SNE
        # tsne_data = np.array(tsne_data)
        # print(tsne_data.shape)
        # with open('../../results/tsne_crash.pkl', 'wb') as tsne_file:
        #     np.save(tsne_file, tsne_data)

    else:
        env.replayer = replayer()
        fuzzer = fuzzing()
        env.fuzzer = fuzzer
        agent = agent_maker()
        tsne_data = []
        min_seq = []
        max_seq = []
        k_storage_state_cvg = {}
        count_of_GMM = 0

        temp_count = 0
        for weather, (start, target), run_name in tqdm.tqdm(env.all_tasks, total=len(list(env.all_tasks))):
            try:
                temp_count += 1
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

                    control, current_entropy, _ = agent.run_step(observations)

                    temp = copy.deepcopy(observations['node'])
                    temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                    vehicle_index = np.nonzero(observations['vehicle'])
                    vehicle_obs = np.zeros(3)
                    vehicle_obs[0] = vehicle_index[0].mean()
                    vehicle_obs[1] = vehicle_index[1].mean()
                    vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                    temp = np.hstack((temp, vehicle_obs))

                    seq_entropy += current_entropy
                    diagnostic = env.apply_control(control)
                    diagnostic.pop("viz_img")
                    diagnostics.append(diagnostic)
                    sequence.append(temp)

                    # HACK: T-SNE
                    # current_tsne = np.array(current_tsne).flatten()
                    # tsne_data.append(current_tsne)

                    if env.is_failure() or env.is_success() or env._tick > 100:
                        result["success"] = env.is_success()
                        result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                        result["total_lights"] = env.traffic_tracker.total_lights
                        result["collided"] = env.collided
                        result["t"] = env._tick
                        break
                print('-----------------------')
                print(total_reward)
                # k_storage_state_cvg = update_dict(k_storage_state_cvg, sequence)
                # print(len(k_storage_state_cvg) / (10000 * 17) * 100)
                first_cvg = env.fuzzer.state_coverage(sequence)
                env.fuzzer.further_mutation((start_pose, env.init_vehicles), rewards=total_reward, entropy=seq_entropy, cvg=first_cvg, original=(start_pose, env.init_vehicles), further_envsetting=[start, target, cu.PRESET_WEATHERS[weather], weather])
            except:
                continue
            if temp_count >= 50:
                break
            # HACK: T_SNE
            # tsne_data = np.array(tsne_data)
            # print(tsne_data.shape)
            # with open('../../results/tsne_random.pkl', 'wb') as tsne_file:
            #     np.save(tsne_file, tsne_data)

        env.first_run = False

        print('fuzzing start!')

        start_fuzz_time = time.time()
        time_of_env = 0
        time_of_fuzzer = 0
        time_of_DynEM = 0
        while len(env.fuzzer.corpus) > 0:
            temp1time = time.time()
            try:
                initial_check = env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
            except:
                initial_check = False
            if initial_check == False:
                print('Trigger initial collision!!!')
                env.fuzzer.drop_current()
                continue

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

                control, current_entropy, _ = agent.run_step(observations)
                temp = copy.deepcopy(observations['node'])
                temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                vehicle_index = np.nonzero(observations['vehicle'])
                vehicle_obs = np.zeros(3)
                vehicle_obs[0] = vehicle_index[0].mean()
                vehicle_obs[1] = vehicle_index[1].mean()
                vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                temp = np.hstack((temp, vehicle_obs))
                seq_entropy += current_entropy
                diagnostic = env.apply_control(control)
                diagnostic.pop("viz_img")
                diagnostics.append(diagnostic)

                sequence.append(copy.deepcopy(temp))

                if env.is_failure() or env.is_success() or env._tick > 100:
                    result["success"] = env.is_success()
                    result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                    result["total_lights"] = env.traffic_tracker.total_lights
                    result["collided"] = env.collided
                    result["t"] = env._tick
                    break
            temp2time = time.time()
            time_of_env += temp2time - temp1time
            cvg = env.fuzzer.state_coverage(sequence)
            time_of_DynEM += time.time() - temp2time
            if env.is_failure() or env.collided:
                env.replayer.store((env.fuzzer.current_pose, env.fuzzer.current_vehicle_info), rewards=total_reward, entropy=seq_entropy, cvg=cvg, original=env.fuzzer.current_original, further_envsetting=env.fuzzer.current_envsetting)
                env.fuzzer.add_crash(env.fuzzer.current_pose)
                print('found: ', len(env.fuzzer.result))
            elif em_guide:
                if total_reward < env.fuzzer.current_reward or cvg < env.fuzzer.GMMthreshold:
                    env.fuzzer.further_mutation((env.fuzzer.current_pose, env.fuzzer.current_vehicle_info), rewards=total_reward, entropy=seq_entropy, cvg=cvg, original=env.fuzzer.current_original, further_envsetting=env.fuzzer.current_envsetting)
            else:
                if total_reward < env.fuzzer.current_reward:
                    env.fuzzer.further_mutation((env.fuzzer.current_pose, env.fuzzer.current_vehicle_info), rewards=total_reward, entropy=seq_entropy, cvg=cvg, original=env.fuzzer.current_original, further_envsetting=env.fuzzer.current_envsetting)

            end_fuzz_time = time.time()
            time_of_fuzzer += end_fuzz_time - temp2time

            print('total reward: ', total_reward, ', coverage: ', cvg, ', passed time: ', end_fuzz_time - start_fuzz_time, ', corpus size: ', len(env.fuzzer.corpus))
            if end_fuzz_time - start_fuzz_time > 3600 * 1:
                break

    print('Total time: ', end_fuzz_time - start_fuzz_time)
    print('Env time: ', time_of_env)
    print('Fuzzer time: ', time_of_fuzzer)
    print('DynEM time: ', time_of_DynEM)

    return result, diagnostics

def save_pickle(replayer):
    corpus = []
    total_crash = len(replayer.corpus)
    for i in range(total_crash):
        single_crash = []
        temp_trans = replayer.corpus[i][0]
        print(temp_trans)
        single_crash.append([temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll])
        temp_vehicleinfo = replayer.corpus[i][1]
        total_vehicle = len(temp_vehicleinfo)
        vehicle_info_crash = []
        for j in range(total_vehicle):
            temp_blue_print = temp_vehicleinfo[j][0]
            temp_trans = temp_vehicleinfo[j][1]
            temp_color = temp_vehicleinfo[j][2]
            temp_vehicle_id = temp_vehicleinfo[j][3]
            vehicle_info_crash.append([temp_blue_print.id, temp_blue_print.tags, temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll, temp_color, temp_vehicle_id])
        corpus.append([single_crash, vehicle_info_crash])
        replayer.envsetting[i][2] = replayer.envsetting[i][3]

    replayer.corpus = corpus
    replayer.original = []
    with open('../../results/noEM_crash.pkl', 'wb') as handle:
        pickle.dump(replayer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Pickle Saved!!!')

def load_pickle(pickle_path, prints):
    with open(pickle_path, 'rb') as handle:
        replayer = pickle.load(handle)
    corpus = []
    envsetting = []
    total_crash = len(replayer.corpus)

    for i in range(total_crash):
        temp_trans = replayer.corpus[i][0][0]
        single_trans = carla.Transform(carla.Location(x=temp_trans[0], y=temp_trans[1], z=temp_trans[2]), carla.Rotation(pitch=temp_trans[3], yaw=temp_trans[4], roll=temp_trans[5]))
        vehicle_info_crash = replayer.corpus[i][1]
        total_vehicle = len(vehicle_info_crash)
        singel_vehicle = []
        for j in range(total_vehicle):
            blue_print = prints.filter(vehicle_info_crash[j][0])[0]
            assert blue_print.tags == vehicle_info_crash[j][1]
            blue_print.set_attribute("role_name", "autopilot")

            color = vehicle_info_crash[j][8]
            vehicle_id = vehicle_info_crash[j][9]
            if color != None:
                blue_print.set_attribute("color", color)
            if vehicle_id != None:
                blue_print.set_attribute("driver_id", vehicle_id)

            trans = carla.Transform(carla.Location(x=vehicle_info_crash[j][2], y=vehicle_info_crash[j][3], z=vehicle_info_crash[j][4]), carla.Rotation(pitch=vehicle_info_crash[j][5], yaw=vehicle_info_crash[j][6], roll=vehicle_info_crash[j][7]))

            singel_vehicle.append((blue_print, trans))

        corpus.append((single_trans, singel_vehicle))

        envsetting.append([replayer.envsetting[i][0], replayer.envsetting[i][1], cu.PRESET_WEATHERS[replayer.envsetting[i][2]], replayer.envsetting[i][3]])
        # replayer.envsetting[i][2] = cu.PRESET_WEATHERS[replayer.envsetting[i][2]]
    replayer.corpus = corpus
    replayer.envsetting = envsetting

    return replayer


def run_benchmark(agent_maker, env, benchmark_dir, seed, resume, max_run=5, replay=False, em_guide=True):
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

        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed, replay, em_guide)

        summary = summary.append(result, ignore_index=True)

        # Do this every timestep just in case.
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)
