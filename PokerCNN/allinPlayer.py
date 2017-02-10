from pypokerengine.players import BasePokerPlayer
from random import randint

# a really dumb PokerPlayer who goes always allin
class AllinPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self):
        self.__community_card = []
        self.__stack = 0
        self.__positionInGameInfos = 0
        self.__last_action = ["call", 0]

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state, dealer):
        # if no simple raise to allin possible (no simple minimal raise or someones overbet you already)
        if valid_actions[2]["amount"]["min"] == -1:
            if valid_actions[1]["amount"] >= self.__stack:
                action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
            else:
                amount = self.__stack - valid_actions[1]["amount"] + int(self.__last_action[1]) \
                         + valid_actions[1]["amount"]
                action = valid_actions[2]["action"]
        # go allin (in the regular way)
        else:
            raise_action_info = valid_actions[2]
            action, amount = raise_action_info["action"], raise_action_info["amount"]["max"]
        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        for i in range(len(game_info["seats"])):
            if game_info["seats"][i]["uuid"] == self.uuid:
                self.__stack = game_info["seats"][i]["stack"]
                self.__positionInGameInfos = i
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.__community_card = []
        self.__last_action = ["call", 0]
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.__community_card = round_state["community_card"]
        self.__stack = round_state["seats"][self.__positionInGameInfos]["stack"]
        if round_state["action_histories"]:
            acthis = round_state["action_histories"]
            action = "call"
            amount = 0
            if "preflop" in acthis:
                for x in acthis["preflop"][::-1]:
                    if self.uuid == x["uuid"]:
                        action = x["action"]
                        amount = x["amount"]
                        break
            if "flop" in acthis:
                for x in acthis["flop"][::-1]:
                    if self.uuid == x["uuid"]:
                        action = x["action"]
                        amount = x["amount"]
                        break
            if "turn" in acthis:
                for x in acthis["turn"][::-1]:
                    if self.uuid == x["uuid"]:
                        action = x["action"]
                        amount = x["amount"]
                        break
            if "river" in acthis:
                for x in acthis["river"][::-1]:
                    if self.uuid == x["uuid"]:
                        action = x["action"]
                        amount = x["amount"]
                        break
            self.__last_action = action, amount
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass