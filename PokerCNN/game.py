from pypokerengine.api.game import setup_config, start_poker
from fishPlayer import FishPlayer
from heuristicPlayer import HeuristicPlayer
from allinPlayer import AllinPlayer
from trainingsGenerator import TrainingsGenerator
from random import randint
import os.path
import glob
import sys
import random
state_to_save = "preflop"
path = "/home/nilus/pokerData/"
    #get the last saved file
def create_bachtes(num, num_players):
    if (os.path.exists(path + state_to_save + "/save0.pickle")):
        files = (glob.glob(path + state_to_save + "/save*.pickle"))
        last_number = []
        for l in range(len(files)):
            last_number.append(int(files[l].split('.pic')[0].split('save')[-1]))
        last_number = max(last_number) + 1
    else:
        last_number = 0
    for i in range(num):
        config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
        config.register_player(name=("p", i), algorithm=AllinPlayer())
        own_seat = randint(1, num_players)
        for x in range(1, own_seat):
            config.register_player(name=("p", x), algorithm=AllinPlayer())
        config.register_player("trainingsGenerator", algorithm=TrainingsGenerator(-1, state_to_save,
                               path, last_number))
        for x in range(own_seat+1, 10):
            config.register_player(name=("p", x), algorithm=AllinPlayer())
        game_result = start_poker(config, verbose = 0)
        # print("I: ", i)
        last_number += 1
