from pypokerengine.api.game import setup_config, start_poker
from fishPlayer import FishPlayer
from heuristicPlayer import HeuristicPlayer
from trainingsGenerator import TrainingsGenerator
import sys

sys.setrecursionlimit(10000000) # 10000 is an example, try with different values

for i in range (100):
    config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
    config.register_player(name="heuristic", algorithm=HeuristicPlayer())
    config.register_player(name="p2", algorithm=HeuristicPlayer())
    '''
    config.register_player(name="p3", algorithm=FishPlayer())
    config.register_player(name="p4", algorithm=FishPlayer())
    config.register_player(name="p5", algorithm=FishPlayer())
    config.register_player(name="p6", algorithm=FishPlayer())
    config.register_player(name="p7", algorithm=FishPlayer())
    config.register_player(name="p8", algorithm=FishPlayer())
    config.register_player(name="trainingsGenerator", algorithm=TrainingsGenerator(-1, "turn"))
    '''
    game_result = start_poker(config, verbose=1)
# print(game_result)