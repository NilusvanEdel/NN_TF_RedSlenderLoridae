from pypokerengine.players import BasePokerPlayer
from random import randint

# a really dumb PokerPlayer who got a 50% chance to either call(check) or raise(minimal)
class FishPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self):
        self.__community_card = []

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, raise_action_info, call_action_info]
        act = randint(0, 1)
        if not act or valid_actions[2]["amount"]["min"] == -1:
            call_action_info = valid_actions[1]
            action, amount = call_action_info["action"], call_action_info["amount"]
        else:
            raise_action_info = valid_actions[2]
            action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]
        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.__community_card = []
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.__community_card = round_state["community_card"]
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass