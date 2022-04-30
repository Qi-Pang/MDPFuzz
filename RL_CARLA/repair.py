import argparse
import time
import warnings
warnings.filterwarnings("ignore")
import sys
# sys.path.append('./carla_RL_IAs/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')

import torchvision.models as models
from pathlib import Path

from benchmark import make_suite, get_suites, ALL_SUITES
from benchmark.run_benchmark import run_benchmark

import torch
import random
import os

from fuzz.fuzz import fuzzing


def run(args, model_path, port, suite, seed, resume, max_run):
    log_dir = model_path

    total_time = 0.0
    for suite_name in get_suites(suite):
        tick = time.time()

        benchmark_dir = log_dir / "benchmark" / ("%s_seed%d" % (suite_name, seed))
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        with make_suite(suite_name, port=port, crop_sky=args.crop_sky) as env:
            from bird_view.models import agent_IAs_RL

            agent_class = agent_IAs_RL.AgentIAsRL
            agent_maker = lambda: agent_class(args)
            run_benchmark(agent_maker, env, benchmark_dir, seed, resume, max_run=max_run)
        elapsed = time.time() - tick
        total_time += elapsed

        print("%s: %.3f hours." % (suite_name, elapsed / 3600))
        break

    print("Total time: %.3f hours." % (total_time / 3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-folder-model",
        required=True,
        type=str,
        help="Folder containing all models, ie the supervised Resnet18 and the RL models",
    )
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--suite", choices=ALL_SUITES, default="town1")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-run", type=int, default=100)

    parser.add_argument(
        "--nb_action_steering",
        type=int,
        default=27,
        help="How much different steering values in the action (should be odd)",
    )
    parser.add_argument(
        "--max_steering", type=float, default=0.6, help="Max steering value possible in action"
    )
    parser.add_argument(
        "--nb_action_throttle",
        type=int,
        default=3,
        help="How much different throttle values in the action",
    )
    parser.add_argument(
        "--max_throttle", type=float, default=1, help="Max throttle value possible in action"
    )

    parser.add_argument("--front-camera-width", type=int, default=288)
    parser.add_argument("--front-camera-height", type=int, default=288)
    parser.add_argument("--front-camera-fov", type=int, default=100)
    parser.add_argument(
        "--crop-sky",
        action="store_true",
        default=False,
        help="if using CARLA challenge model, let sky, we cropped "
        "it for the models trained only on Town01/train weather",
    )

    parser.add_argument("--render", action="store_true", help="Display screen (testing only)")
    parser.add_argument("--disable-cuda", action="store_true", help="disable cuda")
    parser.add_argument("--disable-cudnn", action="store_true", help="disable cuDNN")

    # IQN parameters
    parser.add_argument("--kappa", default=1.0, type=float, help="kappa for Huber Loss in IQN")
    parser.add_argument(
        "--num-tau-samples", default=8, type=int, help="N in equation 3 in IQN paper"
    )
    parser.add_argument(
        "--num-tau-prime-samples", default=8, type=int, help="N' in equation 3 in IQN paper"
    )
    parser.add_argument(
        "--num-quantile-samples", default=32, type=int, help="K in equation 3 in IQN paper"
    )
    parser.add_argument(
        "--quantile-embedding-dim", default=64, type=int, help="n in equation 4 in IQN paper"
    )

    args = parser.parse_args()
    # We take a frame 1 sec before, and the other
    # are 0.2, 0.1 and 0 second before (ORDER IS TAKEN INTO ACCOUNT HERE)
    args.steps_image = [
        -10,
        -2,
        -1,
        0,
    ]
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device("cuda")
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = not args.disable_cudnn
    else:
        args.device = torch.device("cpu")

    args.path_folder_model = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), args.path_folder_model
    )

    run(
        args,
        Path(args.path_folder_model),
        args.port,
        args.suite,
        args.seed,
        args.resume,
        max_run=args.max_run,
    )