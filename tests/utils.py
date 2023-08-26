VECTOR_CONFIG_DICT = {
    "allegro_right": "teleop/allegro_hand_right.yml",
    "allegro_left": "teleop/allegro_hand_left.yml",
    "shadow_right": "teleop/shadow_hand_right.yml",
    "svh_right": "teleop/schunk_svh_hand_right.yml",
}
POSITION_CONFIG_DICT = {
    "allegro_right": "offline/allegro_hand_right.yml",
    # "allegro_left": "offline/allegro_hand_left.yml",
    # "shadow_right": "offline/shadow_hand_right.yml",
    # "svh_right": "offline/schunk_svh_hand_right.yml",
}
DEXPILOT_CONFIG_DICT = {
    "allegro_right": "teleop/allegro_hand_right_dexpilot.yml",
}

ROBOT_NAMES = list(VECTOR_CONFIG_DICT.keys())
