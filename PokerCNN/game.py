from pypokerengine.api.game import setup_config, start_poker
from fishPlayer import FishPlayer
from heuristicPlayer import HeuristicPlayer

config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
config.register_player(name="heuristic", algorithm=HeuristicPlayer())
config.register_player(name="fish", algorithm=FishPlayer())
# error in the implementation the minimal raise is always the big blind!
game_result = start_poker(config, verbose=1)
# print(game_result)