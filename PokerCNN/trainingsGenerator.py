import deuces as de
from random import randint
from pypokerengine.engine.dealer import Dealer
from pypokerengine.api.game import setup_config, start_poker_with_dealer
from heuristicPlayer import HeuristicPlayer

# basically the HeuristicPlayer but this one will store the relevant data to train the neural network
class TrainingsGenerator(HeuristicPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self, next_action):
        self.__community_card = []
        self.__stack = 0
        self.__positionInGameInfos = 0
        self.__last_action = [None]*1
        self.__next_action = next_action

    def declare_action(self, valid_actions, hole_card, round_state, dealer):
        ''' a backup of the current gamestate is created (all stored in the current dealer) and each possible
        round will be played once in this simulation and the result will be stored '''
        if self.__next_action == -1:
            bu_dealer = dealer.copy()
            config = setup_config(max_round=1, initial_stack=100, small_blind_amount=5)
            for i in range (10):
                algorithm = TrainingsGenerator(self.__next_action+i+1)
                bu_dealer.change_algorithm_of_player(self.uuid, algorithm)
                '''
                print("______________________________________________________________________")
                print("simulation_time")
                '''
                game_result = start_poker_with_dealer(config, dealer, verbose=0)
                '''
                print("simulation over")
                print("______________________________________________________________________")
                '''
            return HeuristicPlayer.bot_action(self, valid_actions, hole_card, round_state,
                                        dealer, self.__community_card, self.__stack, self.__last_action)
        else:
            if 0 < self.__next_action < 9:
                action, amount = getAction(valid_actions, self.__stack,
                                           self.__last_action, self.__next_action)
                self.__next_action = - 1
                return action, amount
            else:
                return HeuristicPlayer.bot_action(self, valid_actions, hole_card,
                                round_state, dealer, self.__community_card, self.__stack, self.__last_action)

    def receive_game_start_message(self, game_info):
        for i in range(len(game_info["seats"])):
            if game_info["seats"][i]["uuid"] == self.uuid:
                self.__stack = game_info["seats"][i]["stack"]
                self.__positionInGameInfos = i
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.__community_card = []
        self.__last_action = [None]
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        self.__community_card = round_state["community_card"]
        self.__stack = round_state["seats"][self.__positionInGameInfos]["stack"]
        if round_state["action_histories"]:
            acthis = round_state["action_histories"]
            if "preflop" in acthis:
                count = len(acthis["preflop"])
                for x in acthis["preflop"][::-1]:
                    count -= 1
                    if self.uuid in acthis["preflop"][count]:
                        action = self.uuid in acthis["preflop"][count]["action"]
                        amount = self.uuid in acthis["preflop"][count]["amount"]
                        break
                    break
            if "flop" in acthis:
                count = len(acthis["flop"])
                for x in acthis["flop"][::-1]:
                    count -= 1
                    if self.uuid in acthis["flop"][count]:
                        action = self.uuid in acthis["flop"][count]["action"]
                        amount = self.uuid in acthis["flop"][count]["amount"]
                        break
                    break
            if "turn" in acthis:
                count = len(acthis["turn"])
                for x in acthis["turn"][::-1]:
                    count -= 1
                    if self.uuid in acthis["turn"][count]:
                        action = self.uuid in acthis["turn"][count]["turn"]
                        amount = self.uuid in acthis["turn"][count]["turn"]
                        break
                    break
            if "river" in acthis:
                count = len(acthis["river"])
                for x in acthis["river"][::-1]:
                    count -= 1
                    if self.uuid in acthis["river"][count]:
                        action = self.uuid in acthis["river"][count]["river"]
                        amount = self.uuid in acthis["river"][count]["river"]
                        break
                    break

        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def getAction(valid_actions, stack, last_action, action):
    if action <= 0:
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
    # 45% call or if no raise is possible anymore
    elif action == 1:
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
    # no simple raise possible anymore
    elif valid_actions[2]["amount"]["min"] == -1:
        if valid_actions[1]["amount"] >= stack:
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:
            amount = valid_actions[1]["amount"] - last_action[1] + stack
            action = valid_actions[2]["action"]
    # 10% to raise minimal
    elif action == 2:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]
    # 5% to raise 1,5x
    elif action == 3:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 1.5 + amount_call_action
    # 5% to raise 2x
    elif action == 4:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 2 + amount_call_action
    # 5% to raise 3x
    elif action == 5:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 3 + amount_call_action
    # 5% to raise 5x
    elif action == 6:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 5 + amount_call_action
    # 5% to raise 10x
    elif action == 7:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 10 + amount_call_action
    # 5% to raise 25x
    elif action == 8:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 25 + amount_call_action
    # 5% to go all-in
    else:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["max"]
    if valid_actions[2]["amount"]["max"] != -1:
        if action == "raise" and amount > valid_actions[2]["amount"]["max"]:
            amount = valid_actions[2]["amount"]["max"]
    return action, int(amount)
