
from datetime import datetime
from pytz import timezone
from PokerRL.game._.rl_env.game_rules import HoldemRules

class HandHistoryLogger:
    def __init__(self, logfile, game_type, tablename_type, divisor, output_format):
        self._output_format = output_format
        self._log = open(logfile, "w")
        self._game_type = game_type
        self._tablename_type = tablename_type
        self._divisor = divisor
        self._players = []
        self._winners = []
        self._board = ""
        self._gamestate = -1
        self._butpos = 0
        self._bbpos = 0
        self._sbpos = 0
        self._handcount = 0
        self._actions = []
        self._actions.append("folds")
        self._actions.append("checks")
        self._actions.append("calls")
        self._actions.append("bets")
        self._actions.append("raises")

    def start_hand(self, players, button_pos, sb_pos, bb_pos):
        self._winners.clear()
        self._board = ""
        self._gamestate = -1
        self._handcount += 1
        self._butpos = button_pos
        self._sbpos = sb_pos
        self._bbpos = bb_pos
        timestamp = datetime.now()
        date = timestamp.strftime('%Y/%m/%d')
        time = timestamp.strftime('%H:%M:%S')
        timestamp = datetime.now(timezone("US/Eastern"))
        date_et = timestamp.strftime('%Y/%m/%d')
        time_et = timestamp.strftime('%H:%M:%S')

        self._log.write(f"PokerStars Hand #{self._handcount}:  "
                        f"{self._game_type} - "
                        f"{date} {time} MSK [{date_et} {time_et} ET]\n")

        self._log.write(f"{self._tablename_type} Seat #{button_pos+1} is the button\n")

        # write seats, names and stacks heredown
        for i, p in enumerate(self._players):
            # here we update initial player element that will be appended in future with incoming actions
            # structure is: name, stack, cards, when_folded, won_how_much, pot_type
            p[1] = players[i][1]
            p[2] = None
            p[3] = -1
            p[4] = -1

            self._log.write(f"Seat {i+1}: {self._players[i][0]} "
                            f"(${round(self._players[i][1]/self._divisor, 2)} in chips)\n")

    def set_names(self, names):
        # to change players names according to a current seat plan
        # called when we toss players to imitate button movement
        # before the env.reset, so wehave to create all the players in the list.
        self._players.clear()
        for i, n in enumerate(names):
            # here we create initial player element that will be appended in future with incoming actions
            # structure is: name, stack, cards, when_folded, won_how_much, pot_type
            self._players.append(list((n, None, None, None, None, None)))

    def preflop(self):
        # post it to log!
        self._log.write(f"*** HOLE CARDS ***\n")
        self._gamestate = 0

    def dealt_cards(self, p_id, cards):
        # transform cards to a readable format
        std_cards = []
        for c in cards:
            std_cards.append(HoldemRules.RANK_DICT[c[0]] + HoldemRules.SUIT_DICT[c[1]])

        if len(cards) == 2:
            # print cards only for the very first player - hero
            if p_id == 0:
                self._log.write(f"Dealt to {self._players[p_id][0]} [{std_cards[0]} "
                                f"{std_cards[1]}]\n")
            # update respective players cards
            self._players[p_id][2] = f"{std_cards[0]} {std_cards[1]}"

        elif len(cards) == 4:
            if p_id == 0:
                # print cards only for the very first player - hero
                self._log.write(f"Dealt to {self._players[p_id][0]} [{std_cards[0]} "
                                f"{std_cards[1]} {std_cards[2]} {std_cards[3]}]\n")
            # update respective players cards
            self._players[p_id][2] = f"{std_cards[0]} {std_cards[1]} {std_cards[2]} {std_cards[3]}"

        else:
            raise NotImplementedError("only 2 and 4 hand cards are supported")

    def flop(self, cards):
        std_cards = []
        # transform cards to human-readable form
        for c in cards:
            if c[0] != -127:
                std_cards.append(HoldemRules.RANK_DICT[c[0]] + HoldemRules.SUIT_DICT[c[1]])
        # post it to log!
        self._log.write(f"*** FLOP *** [{std_cards[0]} {std_cards[1]} {std_cards[2]}]\n")
        self._board = f"[{std_cards[0]} {std_cards[1]} {std_cards[2]}]"
        self._gamestate = 1

    def turn(self, cards):
        std_cards = []
        # transform cards to human-readable form
        for c in cards:
            if c[0] != -127:
                std_cards.append(HoldemRules.RANK_DICT[c[0]] + HoldemRules.SUIT_DICT[c[1]])
        # post it to log!
        self._log.write(f"*** TURN *** [{std_cards[0]} {std_cards[1]} {std_cards[2]}] [{std_cards[3]}]\n")
        self._board = f"[{std_cards[0]} {std_cards[1]} {std_cards[2]} {std_cards[3]}]"
        self._gamestate = 2

    def river(self, cards):
        std_cards = []
        # transform cards to human-readable form
        for c in cards:
            if c[0] != -127:
                std_cards.append(HoldemRules.RANK_DICT[c[0]] + HoldemRules.SUIT_DICT[c[1]])
        # post it to log!
        self._log.write(f"*** RIVER *** [{std_cards[0]} {std_cards[1]} {std_cards[2]} "
                        f"{std_cards[3]}] [{std_cards[4]}]\n")
        self._board = f"[{std_cards[0]} {std_cards[1]} {std_cards[2]} {std_cards[3]} {std_cards[4]}]"
        self._gamestate = 3

    def ante_posted(self, p_id, bet):
        self._log.write(f"{self._players[p_id][0]}: posts the ante ${round(bet / self._divisor, 2)}\n")

    def sb_posted(self, p_id, bet):
        self._log.write(f"{self._players[p_id][0]}: posts small blind ${round(bet/self._divisor, 2)}\n")

    def bb_posted(self, p_id, bet):
        self._log.write(f"{self._players[p_id][0]}: posts big blind ${round(bet/self._divisor, 2)}\n")

    def post_action(self, p_id, action, bets_diff=0, bet=0):
        # fold
        if action == 0:
            self._log.write(f"{self._players[p_id][0]}: {self._actions[action]}\n")
            # save street on which the fold occured
            self._players[p_id][3] = self._gamestate
        # check
        elif action == 1:
            self._log.write(f"{self._players[p_id][0]}: {self._actions[action]}\n")
        # call some amount
        elif action == 2:
            self._log.write(f"{self._players[p_id][0]}: {self._actions[action]} ${round(bet/self._divisor, 2)}\n")
        # bet some amount
        elif action == 3:
            self._log.write(f"{self._players[p_id][0]}: {self._actions[action]} ${round(bet/self._divisor, 2)}\n")
        # raise from one to another amount
        elif action == 4:
            self._log.write(f"{self._players[p_id][0]}: {self._actions[action]} "
                            f"${round(bets_diff/self._divisor, 2)} to ${round(bet/self._divisor, 2)}\n")

    def push_winner(self, p_id, amount, cards, pot_type=0):
        # here we add a winner to a list of winners, will be printed in show_down
        winner = []
        winner.append(p_id)
        winner.append(amount)
        winner.append(cards)
        winner.append(pot_type)
        self._winners.append(winner)

    def show_down(self):
        self._log.write(f"*** SHOW DOWN *** \n")
        for p in self._players:
            # if went to showdown and not folded, write his cards
            if p[3] == -1:
                self._log.write(f"{p[0]}: shows [{p[2]}] (a HAND)\n")
        for w in self._winners:
                # if he is also a winner
                # output depends on a pot type, 0 - single pot, 1 - main pot, 2 - side pot
                if w[3] == 0:
                    self._log.write(f"{self._players[w[0]][0]} collected ${round(w[1]/self._divisor, 2)} from pot\n")
                elif w[3] == 1:
                    self._log.write(f"{self._players[w[0]][0]} collected ${round(w[1]/self._divisor, 2)} from main pot\n")
                elif w[3] == 2:
                    self._log.write(f"{self._players[w[0]][0]} collected ${round(w[1]/self._divisor, 2)} from side pot\n")

        #if we had a showdown, then we should have a summary too
        self.summary()

    def no_showdown(self, p_id, amount, pot_uncalled):
        self._log.write(f"Uncalled bet(${round(pot_uncalled/self._divisor, 2)})"
                        f" returned to {self._players[p_id][0]}\n")
        self._log.write(f"{self._players[p_id][0]} collected ${round(amount/self._divisor, 2)} from pot\n")
        self._log.write(f"{self._players[p_id][0]}: doesn't show hand\n")

        # if we had no showdown, still we should have a summary
        self.summary()

    def summary(self):
        self._log.write(f"*** SUMMARY *** \n")

        # add winnings inf to _players to use in summary section
        for p in self._winners:
            # won how much
            self._players[p[0]][4] = p[1]
            # pot type
            self._players[p[0]][5] = p[3]

        # calculate all pots
        tot_pot = 0
        main_pot = 0
        side_pot = 0
        for p in self._players:
            if p[5] == 0:
                tot_pot += p[4]
            elif p[5] == 1:
                main_pot += p[4]
            elif p[5] == 2:
                side_pot += p[4]

        if tot_pot != 0:
            # if no sidepots
            self._log.write(f"Total pot ${round(tot_pot/self._divisor, 2)} | Rake $0\n")
        else:
            # here we have sidepots
            tot_pot = main_pot + side_pot
            self._log.write(f"Total pot ${round(tot_pot / self._divisor, 2)}"
                            f" Main pot ${round(main_pot / self._divisor, 2)}."
                            f" Side pot ${round(side_pot / self._divisor, 2)}. | Rake $0\n")
        if self._board != '':
            self._log.write(f"Board {self._board}\n")

        # next we write player outcomes
        for i, p in enumerate(self._players):

            #create additional string for players who are butt, BB or SB
            pos = ""
            if i == self._butpos:
                pos = " (button)"
            if i == self._sbpos:
                # we use += cuz in HU games button is a small blind as well, so we concatenate it
                pos += " (small blind)"
            elif i == self._bbpos:
                pos = " (big blind)"

            if p[3] == -1:
                # if went to showdown
                if p[4] != -1:
                    # if p has some amount of won chips, e.g. he is a winner
                    self._log.write(f"Seat {i + 1}: {p[0]}{pos} showed [{p[2]}] and won"
                                    f" (${round(p[4] / self._divisor, 2)}) with a HAND\n")
                else:
                    # no won chips, so lost
                    self._log.write(f"Seat {i + 1}: {p[0]}{pos} showed [{p[2]}] and lost with a HAND\n")

            else:
                # if not winner
                if p[3] == 0:
                    self._log.write(f"Seat {i + 1}: {p[0]}{pos} folded before Flop\n")
                if p[3] == 1:
                    self._log.write(f"Seat {i + 1}: {p[0]}{pos} folded on the Flop\n")
                if p[3] == 2:
                    self._log.write(f"Seat {i + 1}: {p[0]}{pos} folded on the Turn\n")
                if p[3] == 3:
                    self._log.write(f"Seat {i + 1}: {p[0]}{pos} folded on the River\n")

        self._log.write(f"\n\n")
