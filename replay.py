import argparse
import time
import pickle
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stats_wrapper import StatsWrapper

def main():
    parser = argparse.ArgumentParser(description='Replay actions in Pokemon Red via Gym environment')
    parser.add_argument('--rom', type=str, help='Path to the Game Boy ROM file', default="./PokemonRed.gb")
    parser.add_argument('--state', type=str, help='Path to the initial state file', default="./has_pokedex_nballs_squirtle.state")
    parser.add_argument('--name', type=str, help='Path to the actions file', default="playthrough.pkl")
    args = parser.parse_args()

    config = {
        "session_path": Path("./session/"),
        "save_final_state": False,
        "print_rewards": False,
        "headless": False,
        "init_state": args.state,
        "action_freq": 24,
        "max_steps": 10280,
        "save_video": False,
        "fast_video": False,
        "gb_path": args.rom,
        "reset_params": {
            "reward_scale": 0.5,
            "event_weight": 4.0,
            "level_weight": 1.0,
            "op_lvl_weight": 0.0,
            "heal_weight": 5.0,
            "explore_weight": 0.1,
            "use_explore_map_obs": True,
            "use_recent_actions_obs": False,
            "zero_recent_actions": False
        }
    }

    # Initialize the environment
    env = StatsWrapper(RedGymEnv(config=config))
    obs, _ = env.reset()
    steps = 0
    rewards = 0

    # Load actions from file
    with open(args.name, "rb") as f:
        actions = pickle.load(f)

    try:
        for action in actions:
            if action == -1:
                continue
            obs, reward, truncated, done, info = env.step(action)
            steps += 1
            rewards += reward

    except KeyboardInterrupt:
        print("Process interrupted, exiting...")

    print(f"Steps taken: {steps}")
    print(f"Return: {rewards}")

    # print info dict iteratively and nicely formatted
    print("Info:")
    info = env.get_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if v > 0:
                    print(f"\t{k}: {v}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
