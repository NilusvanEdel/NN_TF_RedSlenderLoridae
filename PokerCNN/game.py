from pypokerengine.api.game import setup_config, start_poker
from fishPlayer import FishPlayer
from heuristicPlayer import HeuristicPlayer
from trainingsGenerator import TrainingsGenerator

config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
config.register_player(name="heuristic", algorithm=HeuristicPlayer())
config.register_player(name="p2", algorithm=HeuristicPlayer())
# config.register_player(name="fish", algorithm=FishPlayer())
config.register_player(name="trainingsGenerator", algorithm=TrainingsGenerator(-1))
game_result = start_poker(config, verbose=2)
# print(game_result)