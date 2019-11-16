import random
import os
from time import sleep
from math import sqrt

import numpy as np

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

from dql import DQLSolver

game_url = "https://webassembly-game.netlify.com/"

ACTIONS = {0: Keys.SPACE, 1: Keys.ARROW_LEFT, 2: Keys.ARROW_RIGHT}


def build_model(input_size: int, output_size: int) -> object:
    """
    This function initializes the DQL model.

    Parameters:
        - input_size: int - size of input vector
        - output_size: int - size of output vector
    """
    return DQLSolver(input_size, output_size)


def apply_action(decision: int, body: object) -> None:
    """
    This function applies the action to the game window (body).

    Parameters:
        - decision: int - index of action from `ACTIONS`
        - body: object - body element of the game
    """
    body.send_keys(ACTIONS[decision])


def get_state(driver: "webdriver") -> "np.array":
    """
    This function fetch data from the game and parse it to a vector.

    Parameters:
        - driver: webdriver - selenium webdriver object
    """
    response = driver.execute_script(
        """
        return {
            cherry_x: _get_cherry_x(),
            cherry_y: _get_cherry_y(),
            player_x: _get_player_x(),
            player_y: _get_player_y(),
            speed_x: _get_speed_x(),
            speed_y: _get_speed_y(),
        };
    """
    )
    return np.array([list(response.values())]).astype("float32")


def get_game() -> "webdriver":
    """
    This function initialize selenium webdriver and load game site.
    """

    options = Options()
    if "headless" in os.environ.keys():
        options.add_argument("--headless")

    driver = webdriver.Chrome(executable_path="./chromedriver", options=options)
    driver.get(game_url)
    sleep(2)
    return driver


def get_highest_possible_score(driver: "webdriver") -> int:
    """
    This function highest possible score.

    Parameters:
        - driver: webdriver - selenium webdriver object
    """

    return driver.execute_script(
        """
        return _get_highest_possible_score();
    """
    )


def get_score(driver: "webdriver") -> int:
    """
    This function returns current score.

    Parameters:
        - driver: webdriver - selenium webdriver object
    """

    return driver.execute_script(
        """
        return _get_score();
    """
    )


def distance_to_cherry(state: "np.array") -> float:
    """
    This function returns distance from player to cherry.

    Parameters:
        - state: np.array - state array (of shape (6,))
    """
    return sqrt((state[0] - state[2]) ** 2 + (state[1] - state[3]) ** 2)


def reward(prev_state: "np.array", next_state: "np.array", score_delta: int) -> float:
    """
    This function returns reward for transition between states.

    Parameters:
        - prev_state: np.array - state array (of shape (6,))
        - next_state: np.array - state array (of shape (6,))
        - score_delta: int - difference of score between `next_state` and `prev_state`
    """
    if score_delta > 0:
        return 10
    d_a = distance_to_cherry(prev_state)
    d_b = distance_to_cherry(next_state)
    # return np.mean(np.array([d_a, d_b])) * (d_a - d_b) / 1000
    return abs(next_state[4])/100 + abs(next_state[5])/100


def print_training_info(
    run: int, score: int, exploration_rate: float, top_score
) -> None:
    """
    This function prints training info.
    """
    print("-------------------")
    print(f"RUN: {run}")
    print(f"SCORE: {score}")
    print(f"EXPLORATION RATE: {exploration_rate}")
    print(f"TOP SCORE: {top_score}")
    print("-------------------")


def explore_game(epochs: int = 100) -> None:
    """
    This function runs training loop. By default it runs 100 training sessions each consists
    of 15 cherries to collect.

    Parameters:
        - epochs: int - number of trainings
    """

    # Training initialization
    driver = get_game()
    model = build_model(6, 3)
    top_score = 0
    run = 0

    #
    for _ in range(epochs):

        # Initialize new training session
        run += 1
        driver.refresh()
        sleep(2)
        body = driver.find_element_by_tag_name("body")
        current_cherry = 0
        last_score = 0

        for i in range(15):  # For 15 cherries

            while (
                get_highest_possible_score(driver) == current_cherry
            ):  # While cherry tick (3s)
                state = get_state(driver)
                output = model.act(state)
                apply_action(output, body)
                new_score = get_score(driver)
                next_state = get_state(driver)

                score_delta = new_score - last_score
                r = reward(state[0], next_state[0], score_delta)
                last_score = new_score
                print(r)

                model.remember(state, output, r, next_state, bool(score_delta))

                agent_scores_cherry = score_delta > 0
                if agent_scores_cherry:
                    while get_highest_possible_score(driver) == current_cherry:
                        # Skip saving data till next cherry tick
                        continue

            current_cherry = get_highest_possible_score(driver)

        top_score = max([top_score, last_score])
        print_training_info(run, last_score, model.exploration_rate, top_score)
        model.experience_replay()
    model.model.save_weights("model.ckpt")


def play_game():
    """
    This function loads model from `model.ckpt` and runs the game.
    """
    driver = get_game()
    model = build_model(6, 3)
    model.exploration_rate = 0.0
    model.model.load_weights("model.ckpt")
    body = driver.find_element_by_tag_name("body")
    last_score = 0
    while True:
        state = get_state(driver)
        output = model.act(state)
        apply_action(output, body)
        new_score = get_score(driver)
        next_state = get_state(driver)

        score_delta = new_score - last_score
        r = reward(state[0], next_state[0], score_delta)
        print(r)
        last_score = new_score


explore_game()
