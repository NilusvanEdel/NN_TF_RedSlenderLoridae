import deuces as de
from pypokerengine.players import BasePokerPlayer
from random import randint

# our main PokerPlay the taining will be based on him
class HeuristicPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self):
        self.__community_card = []
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        if self.__community_card != []:
            board = [None]*len(self.__community_card)
            hand = [None]*len(hole_card)
            for i in range(len(self.__community_card)):
                card = self.__community_card[i][::-1]
                card = card.replace(card[1], card[1].lower())
                board[i] = de.Card.new(card)
            for i in range(len(hole_card)):
                card = hole_card[i][::-1]
                card = card.replace(card[1], card[1].lower())
                hand[i] = de.Card.new(card)
            action, amount = getPostFlopAction(valid_actions, hand, board)
        else:
            action, amount = getPreFlopAction(valid_actions, hole_card)  # action returned here is sent to the poker engine
        return action, amount

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


def getPreFlopAction(valid_actions, hole_card):
    ''' valid actions in the NN
    0 = check
    1 = fold
    2 = call
    3 = bet
    4 = raise/bet (1x)
    5 = r (1,5x)
    6 = r (2x)
    7 = r (3x)
    8 = r (5x)
    9 = r (10x)
    10 = r (25x)
    11 = r (all-in)
    '''
    # 10% fold baseline
    actionProb = randint(1,100)
    if actionProb <= 10:
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
    # 45% call or if no raise is possible anymore
    elif actionProb > 10 and actionProb <= 55 or valid_actions[2]["amount"]["min"] == -1:
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
    # 10% to raise minimal
    elif actionProb > 55 and actionProb <= 65:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]
    # 5% to raise 1,5x
    elif actionProb > 65 and actionProb <= 70:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*1.5
    # 5% to raise 2x
    elif actionProb > 70 and actionProb <= 75:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*2
    # 5% to raise 3x
    elif actionProb > 75 and actionProb <= 80:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*3
    # 5% to raise 5x
    elif actionProb > 80 and actionProb <= 85:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*5
    # 5% to raise 10x
    elif actionProb > 85 and actionProb <= 90:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*10
    # 5% to raise 25x
    elif actionProb > 90 and actionProb <= 95:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*25
    # 5% to go all-in
    else:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["max"]
    if amount>valid_actions[2]["amount"]["max"]:
        amount = valid_actions[2]["amount"]["max"]
    return action, amount


def getPostFlopAction(valid_actions, hand, board):
    evaluator = de.Evaluator()
    evaluation = evaluator.get_five_card_rank_percentage(evaluator.evaluate(hand, board))
    actionProb = (1-evaluation)*100
    if actionProb <= 10:
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
    # 45% call or if no raise is possible anymore
    elif actionProb > 10 and actionProb <= 55 or valid_actions[2]["amount"]["min"] == -1:
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
    # 10% to raise minimal
    elif actionProb > 55 and actionProb <= 65:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]
    # 5% to raise 1,5x
    elif actionProb > 65 and actionProb <= 70:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*1.5
    # 5% to raise 2x
    elif actionProb > 70 and actionProb <= 75:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*2
    # 5% to raise 3x
    elif actionProb > 75 and actionProb <= 80:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*3
    # 5% to raise 5x
    elif actionProb > 80 and actionProb <= 85:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*5
    # 5% to raise 10x
    elif actionProb > 85 and actionProb <= 90:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*10
    # 5% to raise 25x
    elif actionProb > 90 and actionProb <= 95:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]*25
    # 5% to go all-in
    else:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["max"]
    if amount>valid_actions[2]["amount"]["max"]:
        amount = valid_actions[2]["amount"]["max"]
    return action, amount
