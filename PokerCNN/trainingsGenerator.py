import deuces as de
from random import randint
from pypokerengine.engine.dealer import Dealer
from pypokerengine.api.game import setup_config, start_poker_with_dealer
from heuristicPlayer import HeuristicPlayer
from fishPlayer import FishPlayer
import pickle
import numpy as np
import tensorflow as tf
import os.path
import glob

# basically the HeuristicPlayer but this one will store the relevant data to train the neural network
class TrainingsGenerator(HeuristicPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self, next_action, save_state, path, last_number):
        self.__community_card = []
        self.__stack = 0
        self.__positionInGameInfos = 0
        self.__last_action = ["call", 0]
        # important for recursive call if =-1 it has to save the next moves, otherwise it will perform
        # the given avtion and return the result
        self.__next_action = next_action
        self.__save_state = save_state
        self.__small_blind = 5
        self.__initial_stack = 100
        self.__path = path
        self.__last_number = last_number

    def declare_action(self, valid_actions, hole_card, round_state, dealer):
        ''' a backup of the current gamestate is created (all stored in the current dealer) and each possible
        round will be played once in this simulation and the result will be stored '''
        if self.__next_action == -1: # still needs to save the states
            # check if state to save has come (preflop/flop/ etc.)
            if round_state["action_histories"]:
                acthis = round_state["action_histories"]
                if self.__save_state in acthis:
                    tensor = create_tensor(valid_actions, hole_card, round_state, self.__community_card,
                               self.__small_blind, self.__last_action)
                    # save the current state to file
                    # the state to save
                    state_to_save = self.__save_state
                    with open(self.__path+state_to_save+"/save"+str(self.__last_number)+".pickle", 'wb') as handle:
                        pickle.dump(tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    result_of_moves = [0]*10
                    # backup of the current game_state
                    bu_dealer = dealer.copy()
                    config = setup_config(max_round=1, initial_stack=self.__initial_stack,
                                          small_blind_amount=self.__small_blind)
                    '''
                    print("______________________________________________________________________")
                    print("simulation_time")
                    '''
                    # start the simluation of the 10 moves
                    for i in range(10):
                        # recursive call of trainings generator with iterative increase of next action
                        algorithm = TrainingsGenerator(self.__next_action+i+1, self.__save_state,
                                                       self.__path, self.__last_number)
                        # changes the used algorithm in game to the "new" trainings_gen algorithm
                        bu_dealer.change_algorithm_of_player(self.uuid, algorithm)
                        # play the game
                        game_result = start_poker_with_dealer(config, bu_dealer, verbose=0)
                        # get the result and normalize it
                        amount_win_loss = 0
                        for l in range(len(game_result["players"])):
                            if game_result["players"][l]["uuid"] == self.uuid:
                                amount_win_loss = game_result["players"][l]["stack"] - self.__stack
                        normalized_result = 0.5
                        if amount_win_loss < 0:
                            # normalized for loss (maximum the half of the whole chip size can be lost)
                            # loss is in range from 0 to 0.5
                            normalized_result = amount_win_loss / (self.__stack * 2) + 0.5
                        if amount_win_loss > 0:
                            # normalized for win
                            # win is in range from 0.5 to 1
                            whole_stack = 0
                            for l in range(len(game_result["players"])):
                                whole_stack += game_result["players"][l]["stack"]
                                poss_win = whole_stack - self.__stack
                            normalized_result = amount_win_loss / (poss_win * 2) + 0.5
                        result_of_moves[i] = normalized_result
                    # save the results to file
                    '''
                    print(result_of_moves)
                    print("simulation over")
                    print("______________________________________________________________________")
                    '''
                    with open(self.__path+state_to_save+"/result"+str(self.__last_number)+".pickle", 'wb') as handle:
                        pickle.dump(result_of_moves, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # use the heuristic bot for other actions (if next_action =-1 and state to save not reached yet)
            '''
            return HeuristicPlayer.bot_action(self, valid_actions, hole_card, round_state,
                                        dealer, self.__community_card, self.__stack, self.__last_action)
                                        '''
            return FishPlayer.declare_action(self, valid_actions, hole_card, round_state, dealer)
        # if next=action != -1 and state to save (preflop/flop/...) is reached
        else:
            if 0 <= self.__next_action < 9:
                action, amount = getAction(valid_actions, self.__stack,
                                           self.__last_action, self.__next_action)
                self.__next_action = - 2
                return action, amount
            else:
                return HeuristicPlayer.bot_action(self, valid_actions, hole_card,
                                round_state, dealer, self.__community_card, self.__stack, self.__last_action)
    # initialize the variables with the variables given by the game_engine
    def receive_game_start_message(self, game_info):
        for i in range(len(game_info["seats"])):
            if game_info["seats"][i]["uuid"] == self.uuid:
                self.__stack = game_info["seats"][i]["stack"]
                self.__positionInGameInfos = i
        self.__small_blind = game_info["rule"]["small_blind_amount"]
        self.__initial_stack = game_info["rule"]["initial_stack"]
        pass

    # change the variables in accordance to the new sate (provided by the game engine)
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.__community_card = []
        self.__last_action = ["call", 0]
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    # change the variables in accordance to the new sate (provided by the game engine)
    def receive_game_update_message(self, action, round_state):
        def receive_game_update_message(self, action, round_state):
            self.__community_card = round_state["community_card"]
            self.__stack = round_state["seats"][self.__positionInGameInfos]["stack"]
            " get your last action, important in order to bet correctly, if you bet too much you will simply fold"
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

# create the save_state tensor(will be a np.array in the end)
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
    for i in range(len(round_state["seats"])-1):
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
    # zeropad all of them to 17x17 <-- important that it is a tensor
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
    full_tensor = tf.stack(full_tensor)
    sess = tf.Session()
    with sess.as_default():
        full_tensor = full_tensor.eval()
    return full_tensor


# what is the card written as index of the array above
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


# returns the action as wanted by the game_engine correspondant to your chosen action (0-9)
# important: the game engine wants your actual bet in the end, if you already used to bet e.g. 100 and now need
# to call a bet of 300, it wants {action: call, amount:300} <-- not 200, hence important to keep track of your history
def getAction(valid_actions, stack, last_action, action):
    if action <= 0:
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
    elif action == 1:
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
    # no simple raise possible anymore the game engine provides the amount=-1 for the action raise
    elif valid_actions[2]["amount"]["min"] == -1:
        if valid_actions[1]["amount"] >= stack:
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:
            amount = valid_actions[1]["amount"] - int(last_action[1]) + stack
            action = valid_actions[2]["action"]
    elif action == 2:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]
    elif action == 3:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 1.5 + amount_call_action
    elif action == 4:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 2 + amount_call_action
    elif action == 5:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 3 + amount_call_action
    elif action == 6:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 5 + amount_call_action
    elif action == 7:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 10 + amount_call_action
    elif action == 8:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 25 + amount_call_action
    else:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["max"]
    if valid_actions[2]["amount"]["max"] != -1:
        if action == "raise" and amount > valid_actions[2]["amount"]["max"]:
            amount = valid_actions[2]["amount"]["max"]
    return action, int(amount)
