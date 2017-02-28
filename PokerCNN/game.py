from pypokerengine.api.game import setup_config, start_poker
from fishPlayer import FishPlayer
from heuristicPlayer import HeuristicPlayer
from allinPlayer import AllinPlayer
from trainingsGenerator import TrainingsGenerator
from fishPlayer import FishPlayer
from random import randint
import os.path
import glob
import sys
import random
# which state should be saved (preflop, flop, turn, river)
state_to_save = "river"
# where should the data be saved (notice the folders preflop, flop, turn, river are required beforehand)
path = "/home/nilus/pokerData/"
    #get the last saved file
def create_bachtes(num, num_players):
    # check if their are already existing saves
    if (os.path.exists(path + state_to_save + "/save0.pickle")):
        files = (glob.glob(path + state_to_save + "/save*.pickle"))
        last_number = []
        for l in range(len(files)):
            last_number.append(int(files[l].split('.pic')[0].split('save')[-1]))
        # if so set the the save numer to the highest existing savenumber + 1
        last_number = max(last_number) + 1
    else:
        last_number = 0
    # notice each game lasts only one round and a new game starts. This should be okay, since most cash players
    # usually top their chip amount up if they lost either way (e.g.will always play with a minimum of 50€)
    for i in range(num):
        config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
        # determine the seat of the player randomly
        own_seat = randint(1, num_players)
        for x in range(1, own_seat):
            # the current algorithm(fishPlayer) will always call, hence we will definetly reach the end state
            # of the game if the TrainingsGenerator won't go all-in
            config.register_player(name=("p", x), algorithm=FishPlayer())
        config.register_player("trainingsGenerator", algorithm=TrainingsGenerator(-1, state_to_save,
                               path, last_number))
        for x in range(own_seat+1, num_players+1):
            config.register_player(name=("p", x), algorithm=FishPlayer())
        # if set to 1 the winner of each game will be printed with some additionally information
        game_result = start_poker(config, verbose = 0)
        # print("I: ", i)
        last_number += 1
