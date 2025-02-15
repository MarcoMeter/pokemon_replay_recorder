# Install

`conda create -n poke_replay python=3.11 --yes`

`conda activate poke_replay`

`pip install -r requirements.txt`

# Known Issues

On MacOS, you might need to uninstall pysdl2-dll.

`pip uninstall pysdl2-dll`

# Add ROM

The Pokémon Red Rom should be located at the path `./PokemonRed.gb`. Otherwise specify the path using `--rom my_rom.gb`.

# Record playthrough

`python play.py --name my_replay.json`

Controls:

```
action_mapping = {
    pygame.K_UP: 3,
    pygame.K_DOWN: 0,
    pygame.K_LEFT: 1,
    pygame.K_RIGHT: 2,
    pygame.K_a: 4, # A
    pygame.K_s: 5, # B
    pygame.K_RETURN: 6,
}
```

Note that the key A is used for `A` and the key S is used for `B`. Change this to your likings. I did this because of the different local keyboard layouts.

When done recording, press `ESC`, `Ctrl + C`, or just quit.

You may also resume a saved playthrough as follows:

`python play.py --name my_replay_resume.json --resume my_replay.json`

During the playthrough, you can press `P` to capture a screenshot.

# Replay

`python replay.py --name example_replay.json`

or headless

`python replay.py --name example_replay.json --headless`

# Recording Instructions

- Play untill receiving TM Dig (ensure to beat Misty and get Badge 2 as well)
- No need to speed run
- No need to play absolutely precise
- Your strategy can be pretty free after all
- Out of scope:
    - Completing the pokedex
    - Using the storage system
- (Optional: You can proceed further in the main story if you want to keep recording.) 
