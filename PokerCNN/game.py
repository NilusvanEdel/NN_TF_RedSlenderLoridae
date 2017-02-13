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
path = "/home/nilus/PycharmProjects/pokerData/"
    #get the last saved file
if (os.path.exists(path + state_to_save + "/save0.pickle")):
    files = (glob.glob(path + state_to_save + "/save*.pickle"))
    print("dafuq: ", files)
    last_number = []
    for l in range(len(files)):
        last_number.append(int(files[l].split('.pic')[0].split('save')[-1]))
    last_number = max(last_number) + 1
else:
    last_number = 0
for i in range(200):
    config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
    '''
    config.register_player(name="heuristic", algorithm=HeuristicPlayer())
    config.register_player(name="p2", algorithm=HeuristicPlayer())
    config.register_player(name="p3", algorithm=FishPlayer())
    config.register_player(name="p4", algorithm=FishPlayer())
    config.register_player(name="p5", algorithm=FishPlayer())
    config.register_player(name="p6", algorithm=FishPlayer())
    config.register_player(name="p7", algorithm=FishPlayer())
    '''
    if randint(1,2) == 1:
        config.register_player(name="p8", algorithm=AllinPlayer())
        config.register_player(name="trainingsGenerator", algorithm=TrainingsGenerator(-1, state_to_save,
                                                                                       path, last_number))
    else:
        config.register_player(name="trainingsGenerator", algorithm=TrainingsGenerator(-1, state_to_save,
                                                                                       path, last_number))
        config.register_player(name="p8", algorithm=AllinPlayer())
    game_result = start_poker(config, verbose = 0)
    print("I: ", i)
    last_number += 1