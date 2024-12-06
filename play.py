import argparse
import pickle
from pathlib import Path
import pygame
from red_gym_env_v2 import RedGymEnv
import numpy as np

def process_frame(frame):
    frame = frame.transpose((1, 0, 2))
    frame = frame.squeeze()
    frame = np.stack((frame, frame, frame), axis=-1)
    return frame

def update_screen(screen, frame, screen_width, screen_height):
    obs_surface = pygame.surfarray.make_surface(frame)
    obs_surface = pygame.transform.scale(obs_surface, (screen_width, screen_height))
    screen.blit(obs_surface, (0, 0))
    pygame.display.flip()

def main():
    parser = argparse.ArgumentParser(description='Play Pokemon Red via Gym environment')
    parser.add_argument('--rom', type=str, help='Path to the Game Boy ROM file', default="./PokemonRed.gb")
    parser.add_argument('--state', type=str, help='Path to the initial state file', default="./has_pokedex_nballs_squirtle.state")
    parser.add_argument('--name', type=str, help='Name of the playthrough', default="playthrough.pkl")
    args = parser.parse_args()

    config = {
        "session_path": Path("./session/"),
        "save_final_state": False,
        "print_rewards": False,
        "headless": True,
        "init_state": args.state,
        "action_freq": 24,
        "max_steps": 10280, # is incremented by 2048 for each completed event
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
    env = RedGymEnv(config=config)
    obs, _ = env.reset()

    # Initialize Pygame
    pygame.init()
    scale_factor = 10
    screen_width, screen_height = obs["screens"].shape[1] * scale_factor, obs["screens"].shape[0] * scale_factor
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Pokemon Red Playthrough')
    clock = pygame.time.Clock()

    # Keyboard controls
    action_mapping = {
        pygame.K_UP: 3,
        pygame.K_DOWN: 0,
        pygame.K_LEFT: 1,
        pygame.K_RIGHT: 2,
        pygame.K_a: 4, # A
        pygame.K_s: 5, # B
        pygame.K_RETURN: 6,
    }

    # Record actions of the playthrough
    actions = []
    debounce_time = 0.1  # 100 ms
    last_action_time = 0

    # Press `B` as initial dummy action and step the environment
    actions.append(5)
    obs, reward, _, done, info = env.step(5)
    # Render the environment using pygame
    frame = process_frame(env.render(reduce_res=False))
    scale_factor = 1
    update_screen(screen, frame, screen_width, screen_height)

    try:
        done = False
        while not done:
            # Check for pressed keys and set the corresponding action
            action = -1
            current_time = pygame.time.get_ticks() / 1000  # Convert to seconds
            if current_time - last_action_time > debounce_time:
                keys = pygame.key.get_pressed()
                for key, mapped_action in action_mapping.items():
                    if keys[key]:
                        action = mapped_action
                        actions.append(action)
                        last_action_time = current_time
                        break

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    done = True

            # If no valid action is pressed, continue
            if action == -1:
                continue

            # Step the environment
            obs, reward, _, done, info = env.step(action)

            # Render the environment using pygame
            frame = process_frame(env.render(reduce_res=False))
            update_screen(screen, frame, screen_width, screen_height)

            # Control the frame rate
            clock.tick(12)

    except KeyboardInterrupt:
        print("Process interrupted, exiting...")

    finally:
        pygame.quit()

    # Save the actions
    with open(args.name, "wb") as f:
        pickle.dump(actions, f)

if __name__ == "__main__":
    main()
