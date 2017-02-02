import deuces as de
from pypokerengine.players import BasePokerPlayer
from random import randint

# our main PokerPlay the training will be based on him (thus he has only 12 possible actions)
class HeuristicPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self):
        self.__community_card = []
        self.__stack = 0
        self.__positionInGameInfos = 0
        self.__last_action = ["call", 0]
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state, dealer):
        return HeuristicPlayer.bot_action(self, valid_actions, hole_card,
                            round_state, dealer, self.__community_card, self.__stack, self.__last_action)

    def bot_action(self, valid_actions, hole_card, round_state, dealer, community_card, stack, last_action):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        if community_card:
            board = [None] * len(community_card)
            hand = [None] * len(hole_card)
            for i in range(len(community_card)):
                card = community_card[i][::-1]
                card = card.replace(card[1], card[1].lower())
                board[i] = de.Card.new(card)
            for i in range(len(hole_card)):
                card = hole_card[i][::-1]
                card = card.replace(card[1], card[1].lower())
                hand[i] = de.Card.new(card)
            action, amount = getPostFlopAction(valid_actions, hand, board, stack, last_action)
        else:
            action, amount = getPreFlopAction(valid_actions, hole_card, stack, last_action)
            stack -= amount
        return action, amount

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


def getPreFlopAction(valid_actions, hole_card, stack, last_action):
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
    actionProb = randint(1, 100)
    if actionProb <= 10:
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
    # 45% call or if no raise is possible anymore
    elif actionProb > 10 and actionProb <= 55:
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
    # if no simple raise is possible anymore
    elif valid_actions[2]["amount"]["min"] == -1:
        if valid_actions[1]["amount"] >= stack:
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:
            # no idea why there is sometimes a failure otherwise
            amount = valid_actions[1]["amount"] - int(last_action[1]) + stack
            action = valid_actions[2]["action"]
    # 10% to raise minimal
    elif actionProb > 55 and actionProb <= 65:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]
    # 5% to raise 1,5x
    elif actionProb > 65 and actionProb <= 70:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 1.5 + amount_call_action
    # 5% to raise 2x
    elif actionProb > 70 and actionProb <= 75:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 2 + amount_call_action
    # 5% to raise 3x
    elif actionProb > 75 and actionProb <= 80:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 3 + amount_call_action
    # 5% to raise 5x
    elif actionProb > 80 and actionProb <= 85:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 5 + amount_call_action
    # 5% to raise 10x
    elif actionProb > 85 and actionProb <= 90:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action)* 10 + amount_call_action
    # 5% to raise 25x
    elif actionProb > 90 and actionProb <= 95:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action)* 25 + amount_call_action
    # 5% to go all-in
    else:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["max"]
    if valid_actions[2]["amount"]["max"] != -1:
        if action == "raise" and amount > valid_actions[2]["amount"]["max"]:
            amount = valid_actions[2]["amount"]["max"]
    return action, int(amount)


def getPostFlopAction(valid_actions, hand, board, stack, last_action):
    evaluator = de.Evaluator()
    evaluation = evaluator.get_five_card_rank_percentage(evaluator.evaluate(hand, board))
    actionProb = (1-evaluation)*100
    if actionProb <= 10:
        fold_action_info = valid_actions[0]
        action, amount = fold_action_info["action"], fold_action_info["amount"]
    # 45% call or if no raise is possible anymore
    elif actionProb > 10 and actionProb <= 55:
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
    # no simple raise possible anymore
    elif valid_actions[2]["amount"]["min"] == -1:
        if valid_actions[1]["amount"] >= stack:
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:
            # no idea why necessary
            amount = valid_actions[1]["amount"] - int(last_action[1]) + stack
            action = valid_actions[2]["action"]
    # 10% to raise minimal
    elif actionProb > 55 and actionProb <= 65:
        raise_action_info = valid_actions[2]
        action, amount = raise_action_info["action"], raise_action_info["amount"]["min"]
    # 5% to raise 1,5x
    elif actionProb > 65 and actionProb <= 70:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 1.5 + amount_call_action
    # 5% to raise 2x
    elif actionProb > 70 and actionProb <= 75:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 2 + amount_call_action
    # 5% to raise 3x
    elif actionProb > 75 and actionProb <= 80:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 3 + amount_call_action
    # 5% to raise 5x
    elif actionProb > 80 and actionProb <= 85:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 5 + amount_call_action
    # 5% to raise 10x
    elif actionProb > 85 and actionProb <= 90:
        raise_action_info = valid_actions[2]
        amount_call_action = valid_actions[1]["amount"]
        action = raise_action_info["action"]
        amount = (raise_action_info["amount"]["min"] - amount_call_action) * 10 + amount_call_action
    # 5% to raise 25x
    elif actionProb > 90 and actionProb <= 95:
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
