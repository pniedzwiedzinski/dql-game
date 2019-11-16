import random
import csv
import numpy as np
import tensorflow as tf
from time import sleep
from math import sqrt
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from dql import DQLSolver

game_url = "https://webassembly-game.netlify.com/"

ACTIONS = {0: Keys.SPACE, 1: Keys.ARROW_LEFT, 2: Keys.ARROW_RIGHT}


def build_model(input_size, output_size):
    return DQLSolver(input_size, output_size)


def send_keys(decision, body):
    body.send_keys(ACTIONS[decision])


def get_state(driver):
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


def get_game():
    driver = webdriver.Chrome(executable_path="./chromedriver")
    driver.get(game_url)
    sleep(2)
    body = driver.find_element_by_tag_name("body")
    return driver, body


def get_max_cherry(driver):
    return driver.execute_script(
        """
        return _get_highest_possible_score();
    """
    )


def get_score(driver):
    return driver.execute_script(
        """
        return _get_score();
    """
    )


def distance_to_cherry(state):
    return sqrt((state[0] - state[2]) ** 2 + (state[1] - state[3]) ** 2)


def reward(prev_state, next_state, score_delta):
    if score_delta > 0:
        return 30000
    d_a = distance_to_cherry(prev_state)
    d_b = distance_to_cherry(next_state)
    return  np.mean(np.array([d_a, d_b])) * (d_a - d_b) / 1000


def play_game():
    driver, _ = get_game()
    top_score = 0
    for _ in range(100):
        driver.refresh()
        sleep(2)
        body = driver.find_element_by_tag_name("body")
        model = build_model(6, 3)

        current_cherry = 0
        last_score = 0

        for i in range(15):
            while get_max_cherry(driver) == current_cherry:
                state = get_state(driver)
                output = model.act(state)
                send_keys(output, body)
                new_score = get_score(driver)
                next_state = get_state(driver)

                score_delta = new_score - last_score
                r = reward(state[0], next_state[0], score_delta)
                last_score = new_score
                print(r)

                model.remember(state, output, r, next_state, bool(score_delta))

                if score_delta > 0:
                    while get_max_cherry(driver) == current_cherry:
                        # Skip saving data till next cherry tick
                        continue

            current_cherry = get_max_cherry(driver)

        top_score = max([top_score, last_score])
        print("-------------------")
        print("TOP SCORE: " + str(top_score))
        print("-------------------")
        model.experience_replay()


play_game()