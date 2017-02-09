import deuces as de
from random import randint
from pypokerengine.engine.dealer import Dealer
from pypokerengine.api.game import setup_config, start_poker_with_dealer
from heuristicPlayer import HeuristicPlayer
import pickle
import numpy as np
import tensorflow as tf

# basically the HeuristicPlayer but this one will store the relevant data to train the neural network
class TrainingsGenerator(HeuristicPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self, next_action, save_state):
        self.__community_card = []
        self.__stack = 0
        self.__positionInGameInfos = 0
        self.__last_action = ["call", 0]
        self.__next_action = next_action
        self.__save_state = save_state
        self.__small_blind = 5
        self.__initial_stack = 100

    def declare_action(self, valid_actions, hole_card, round_state, dealer):
        ''' a backup of the current gamestate is created (all stored in the current dealer) and each possible
        round will be played once in this simulation and the result will be stored '''
        if self.__next_action == -1:
            bu_dealer = dealer.copy()
            config = setup_config(max_round=1, initial_stack=self.__initial_stack,
                                  small_blind_amount=self.__small_blind)
            if round_state["action_histories"]:
                acthis = round_state["action_histories"]
                if self.__save_state in acthis:
                    tensor = create_tensor(valid_actions, hole_card, round_state, self.__community_card,
                               self.__small_blind, self.__last_action)
                    result_of_moves = [0]*10
                    for i in range(10):
                        algorithm = TrainingsGenerator(self.__next_action+i+1, self.__save_state)
                        bu_dealer.change_algorithm_of_player(self.uuid, algorithm)
                        '''
                        print("______________________________________________________________________")
                        print("simulation_time")
                        '''
                        game_result = start_poker_with_dealer(config, dealer, verbose=0)
                        amount_win_loss = 0
                        for l in range(len(game_result["players"])):
                            if game_result["players"][l]["uuid"] == self.uuid:
                                if self.__stack > game_result["players"][l]["stack"]:
                                    amount_win_loss = self.__stack - game_result["players"][l]["stack"]
                                else:
                                    amount_win_loss += game_result["players"][l]["stack"] - self.__stack
                        normalized_result = 0.5
                        if amount_win_loss < 0:
                            # normalized for loss (maximum the half of the whole chip size can be lost)
                            # loss is in range from 0 to 0.5
                            normalized_result = amount_win_loss / self.__initial_stack * len(game_result["players"]) / 2
                        if amount_win_loss > 0:
                            # normalized for win
                            # win is in range from 0.5 to 1
                            whole_stack = 0
                            for l in range(len(game_result["players"])):
                                whole_stack += game_result["players"][l]["stack"]
                            normalized_result = amount_win_loss / whole_stack + 0.5
                        # todo serialize this
                        result_of_moves[i] = normalized_result
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
                self.__next_action = - 2
                return action, amount
            else:
                return HeuristicPlayer.bot_action(self, valid_actions, hole_card,
                                round_state, dealer, self.__community_card, self.__stack, self.__last_action)

    def receive_game_start_message(self, game_info):
        for i in range(len(game_info["seats"])):
            if game_info["seats"][i]["uuid"] == self.uuid:
                self.__stack = game_info["seats"][i]["stack"]
                self.__positionInGameInfos = i
        self.__small_blind = game_info["rule"]["small_blind_amount"]
        self.__initial_stack = game_info["rule"]["initial_stack"]
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
        action = "call"
        amount = "0"
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
                        action = self.uuid in acthis["turn"][count]["action"]
                        amount = self.uuid in acthis["turn"][count]["amount"]
                        break
                    break
            if "river" in acthis:
                count = len(acthis["river"])
                for x in acthis["river"][::-1]:
                    count -= 1
                    if self.uuid in acthis["river"][count]:
                        action = self.uuid in acthis["river"][count]["action"]
                        amount = self.uuid in acthis["river"][count]["amount"]
                        break
                    break
            self.__last_action = action, amount
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def create_tensor(valid_actions, hole_card, round_state, community_card, small_blind, last_action):
    # write cards in array
    ind_of_card_one = get_index_of_card(hole_card[0])
    ind_of_card_two = get_index_of_card(hole_card[1])
    hole_cards_arr = np.zeros(shape=(13, 4))
    hole_cards_arr[ind_of_card_one[1], ind_of_card_one[0]] = 1
    hole_cards_arr[ind_of_card_two[1], ind_of_card_two[0]] = 1
    flop_cards_arr = np.zeros(shape=(13, 4))
    turn_cards_arr = np.zeros(shape=(13, 4))
    river_cards_arr = np.zeros(shape=(13, 4))
    for i in range (len(community_card)):
        ind_of_card = get_index_of_card(community_card[i])
        if (i < 3):
            flop_cards_arr[ind_of_card[1], ind_of_card[0]] = 1
        elif (i < 4):
            turn_cards_arr[ind_of_card[1], ind_of_card[0]] = 1
        else:
            river_cards_arr[ind_of_card[1], ind_of_card[0]] = 1
    # all cards written in one single array
    all_cards = hole_cards_arr + flop_cards_arr + turn_cards_arr + river_cards_arr
    # mainpot numerical coded dependant on big blind (= 2*small_blind)
    mainpot = round_state["pot"]["main"]["amount"]
    mainpot_arr = np.zeros(shape=(13, 4))
    mainpot_in_bb = mainpot // (small_blind*2)
    # everything above 52 equals allin to the algorithm
    if mainpot_in_bb > 52:
        mainpot_in_bb= 52
    mainpot_cols = mainpot_in_bb // 4
    if mainpot_cols > 13:
        mainpot_cols = 13
    for i in range(mainpot_cols):
        mainpot_arr[i][:] = 1
    if mainpot_cols == 0 and mainpot_in_bb > 0:
        rest = mainpot_in_bb
    elif mainpot_cols == 0:
        rest = 0
    else:
        rest = mainpot_in_bb % mainpot_cols
    for i in range(rest):
        mainpot_arr[mainpot_cols][i] = 1
    # players active in array
    players_active_arr = np.zeros(shape=(9, 9))
    for i in range(len(round_state["seats"])):
        if round_state["seats"][i]["state"] == "participating" or round_state["seats"][i]["state"] == "allin":
            players_active_arr[i][:] = 1
    # dealer button in array
    dealer_btn_arr = np. zeros(shape=(9, 9))
    dealer_btn_arr[:][round_state["dealer_btn"]] = 1
    # amount needed to call coded numerical dependant on sb
    amount = valid_actions[1]["amount"] - int(last_action[1])
    amount_to_call_arr = np.zeros(shape=(13, 4))
    amount_to_call = amount // small_blind
    amount_to_call_cols = amount_to_call // 4
    for i in range(amount_to_call_cols):
        amount_to_call_arr[:][i] = 1
    rest = 0
    if (amount_to_call_cols == 0 and amount // small_blind > 0):
        rest = amount_to_call
    elif amount_to_call_cols == 0:
        rest = 0
    else:
        rest = amount % amount_to_call_cols
    for i in range(rest):
        amount_to_call_arr[amount_to_call_cols][i] = 1
    # convert the arrays to tensors
    tensor1 = tf.convert_to_tensor(hole_cards_arr)
    tensor2 = tf.convert_to_tensor(flop_cards_arr)
    tensor3 = tf.convert_to_tensor(turn_cards_arr)
    tensor4 = tf.convert_to_tensor(river_cards_arr)
    tensor5 = tf.convert_to_tensor(all_cards)
    tensor6 = tf.convert_to_tensor(amount_to_call_arr)
    tensor7 = tf.convert_to_tensor(mainpot_arr)
    tensor8 = tf.convert_to_tensor(dealer_btn_arr)
    tensor9 = tf.convert_to_tensor(players_active_arr)
    # zeropad all of them to 17x17
    tensor1 = tf.pad(tensor1, [[2, 2], [6, 7]], "CONSTANT")
    tensor2 = tf.pad(tensor2, [[2, 2], [6, 7]], "CONSTANT")
    tensor3 = tf.pad(tensor3, [[2, 2], [6, 7]], "CONSTANT")
    tensor4 = tf.pad(tensor4, [[2, 2], [6, 7]], "CONSTANT")
    tensor5 = tf.pad(tensor5, [[2, 2], [6, 7]], "CONSTANT")
    tensor6 = tf.pad(tensor6, [[2, 2], [6, 7]], "CONSTANT")
    tensor7 = tf.pad(tensor7, [[2, 2], [6, 7]], "CONSTANT")
    tensor8 = tf.pad(tensor8, [[4, 4], [4, 4]], "CONSTANT")
    tensor9 = tf.pad(tensor9, [[4, 4], [4, 4]], "CONSTANT")
    # create a 9x17x17 of all of them
    full_tensor = [tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7, tensor8, tensor9]
    full_tensor = tf.pack(full_tensor)
    return full_tensor

def get_index_of_card(card):
    ind_of_card = [0, 0]
    if card[0] == "S":
        ind_of_card[0] = 1
    elif card[0] == "H":
        ind_of_card[0] = 2
    elif card[0] == "D":
        ind_of_card[0] = 3
    if card[1] == "A":
        ind_of_card[1] = 12
    elif card[1] == "K":
        ind_of_card[1] = 11
    elif card[1] == "Q":
        ind_of_card[1] = 10
    elif card[1] == "J":
        ind_of_card[1] = 9
    elif card[1] == "T":
        ind_of_card[1] = 8
    else:
        ind_of_card[1] = int(card[1])-2
    return ind_of_card


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
            amount = valid_actions[1]["amount"] - int(last_action[1]) + stack
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
