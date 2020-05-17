# Copyright (c) 2019 Eric Steinberger


import ctypes
import os
from os.path import join as ospj

import itertools
import numpy as np

from PokerRL._.CppWrapper import CppWrapper
from PokerRL.game._.rl_env.game_rules import HoldemRules


class CppHandeval(CppWrapper):

    def __init__(self):
        super().__init__(path_to_dll=ospj(os.path.dirname(os.path.realpath(__file__)),
                                          "lib_hand_eval." + self.CPP_LIB_FILE_ENDING))
        self._clib.get_hand_rank_52_holdem.argtypes = [
            self.ARR_2D_ARG_TYPE,
            self.ARR_2D_ARG_TYPE
        ]
        self._clib.get_hand_rank_52_holdem.restype = ctypes.c_int32

        self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem.argtypes = [
            self.ARR_2D_ARG_TYPE,
            self.ARR_2D_ARG_TYPE,
            ctypes.c_int32,
            self.ARR_2D_ARG_TYPE,
            self.ARR_2D_ARG_TYPE
        ]
        self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem.restype = None

    def get_hand_rank_52_holdem(self, hand_2d, board_2d):
        """
        Args:
            hand_2d (np.ndarray(shape=[5,2], dtype=int8)):      [rank, suit], [rank, suit]]
            board_2d (np.ndarray(shape=[5,2], dtype=int8)):     [rank, suit], [rank, suit], ...]

        Returns:
            int: integer representing strength of the strongest 5card hand in the 7 cards. higher is better.
        """
        return self._clib.get_hand_rank_52_holdem(self.np_2d_arr_to_c(hand_2d), self.np_2d_arr_to_c(board_2d))

    def get_hand_rank_52_plo(self, hand_2d, board_2d):
        """
        Args:
            hand_2d (np.ndarray(shape=[5,2], dtype=int8)):      [rank, suit], [rank, suit]]
            board_2d (np.ndarray(shape=[5,2], dtype=int8)):     [rank, suit], [rank, suit], ...]

        Returns:
            int: integer representing strength of the strongest 5card hand
             in the 4 player + 5 board card, given that player have to use any 2. higher is better.
        Works:
            computes hand strength for all 6 possible combs of given Omaha hand, returns highest
            ---NEXT recreate this shit in pure C with fast 7-card evaluate algo(now its naive algo ~50k evals/sec)
            now it is totaling for a PLO hand at 7k hands/sec, while it can be around N mil hands/sec
        """
        hand_2c_combs = np.empty([6,2,2],dtype=np.int8)
        idxs = np.array([0,1,2,3])
        dt = np.dtype([('', idxs.dtype)] * 2)
        # fill array right from combinations() func iteratively - its fast!
        idxs_combs = np.fromiter(itertools.combinations(idxs, 2), dt)
        for n, i in enumerate(idxs_combs):
           hand_2c_combs[n,0] = hand_2d[i[0]]
           hand_2c_combs[n,1] = hand_2d[i[1]]
        maxres = -1
        for hand in hand_2c_combs:
            res = self._clib.get_hand_rank_52_holdem(self.np_2d_arr_to_c(hand), self.np_2d_arr_to_c(board_2d))
            if res > maxres:
                maxres = res
        return maxres


    def get_hand_rank_all_hands_on_given_boards_52_holdem(self, boards_1d, lut_holder):
        """
        Args:
            boards_1d (np.ndarray(shape=[N, 5], dtype=int8)):   [[c1, c2, c3, c4, c5], [c1, c2, .., c5], ...}

        Returns:
            np.ndarray(shape=[N, RANGE_SIZE], dtype=int32):     hand_rank for each possible hand; -1 for
                                                                blocked on each of the given boards

        """
        assert len(boards_1d.shape) == 2
        assert boards_1d.shape[1] == 5
        hand_ranks = np.full(shape=(boards_1d.shape[0], HoldemRules.RANGE_SIZE), fill_value=-1, dtype=np.int32)
        self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem(
            self.np_2d_arr_to_c(hand_ranks),  # int32**
            self.np_2d_arr_to_c(boards_1d),  # int8**
            boards_1d.shape[0],  # int (number of boards)
            self.np_2d_arr_to_c(lut_holder.LUT_IDX_2_HOLE_CARDS),  # int8**
            self.np_2d_arr_to_c(lut_holder.LUT_1DCARD_2_2DCARD)  # int8**
        )
        return hand_ranks
