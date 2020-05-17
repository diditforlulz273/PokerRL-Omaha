# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
from PokerRL.eval.lbr import _util
from PokerRL.game.games import DiscretizedNLHoldem, PLO
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from PokerRL.game import bet_sets
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile


class TestCppLib(TestCase):
    @classmethod
    def setUpClass(cls):
        t_prof = TrainingProfile(name="NLH_EXPLOITABILITY_PLO",
                                 nn_type="feedforward",

                                 DISTRIBUTED=False,
                                 CLUSTER=False,
                                 n_learner_actor_workers=2,  # 20 workers

                                 max_buffer_size_adv=1e6,
                                 max_buffer_size_avrg=1e6,
                                 export_each_net=False,
                                 checkpoint_freq=8,
                                 eval_agent_export_freq=4,  # produces GBs!

                                 # How many actions out of all legal on current step to branch randomly = action bredth limit
                                 n_actions_traverser_samples=4,  # 3 is the default, 4 is current max for b_2
                                 # number of traversals gives some amount of otcomes to train network on
                                 # mult = 1...4, buffer appends every() step with new data
                                 n_traversals_per_iter=30,
                                 # number of mini_batch fetches and model updates on each step
                                 n_batches_adv_training=801,  # 1024
                                 n_batches_avrg_training=2048,  # 2048

                                 use_pre_layers_adv=True,
                                 n_cards_state_units_adv=192,
                                 n_merge_and_table_layer_units_adv=64,
                                 n_units_final_adv=64,

                                 # amount of batch to feed to NN at once, fetched from buffer randomly.
                                 mini_batch_size_adv=512,  # 256
                                 mini_batch_size_avrg=512,  # 512
                                 init_adv_model="random",  # warm start neural weights with init from last iter
                                 init_avrg_model="random",
                                 # use_pre_layers_avrg=False,  # shallower nets

                                 lr_avrg=0.001,
                                 game_cls=DiscretizedNLHoldem,  # PLO or DiscretizedNLHoldem
                                 env_bldr_cls=VanillaEnvBuilder,
                                 agent_bet_set=bet_sets.PL_2,
                                 n_seats=2,
                                 start_chips=10000,

                                 # You can specify one or both modes. Choosing both is useful to compare them.
                                 eval_modes_of_algo=(
                                     EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                     # EvalAgentDeepCFR.EVAL_MODE_AVRG_NET
                                 ),

                                 # enables simplified obs. Default works also for 3+ players
                                 use_simplified_headsup_obs=True,

                                 log_verbose=True,
                                 lbr_args=LBRArgs(lbr_bet_set=bet_sets.PL_2,
                                                  n_lbr_hands_per_seat=100,
                                                  lbr_check_to_round=Poker.TURN,
                                                  # recommended to set to Poker.TURN for 4-round games.
                                                  n_parallel_lbr_workers=1,
                                                  use_gpu_for_batch_eval=False,
                                                  DISTRIBUTED=True,
                                                  ),
                                 )

        cls._eval_env_bldr = _util.get_env_builder_lbr(t_prof=t_prof)
        stk = [10000,10000]
        cls._env = cls._eval_env_bldr.get_new_env(is_evaluating=True, stack_size=stk)
        cls.t_prof = t_prof

    def test_get_hand_rank_52_holdem(self):
        cpp_poker = CppHandeval()
        b = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
        h = np.array([[11, 3], [5, 1]], dtype=np.int8)
        handstr = cpp_poker.get_hand_rank_52_holdem(hand_2d=h, board_2d=b)
        assert handstr == 1240554

    def test_get_hand_rank_52_plo(self):
        cpp_poker = CppHandeval()
        b = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
        h = np.array([[11, 3], [5, 1], [5,2], [11,0]], dtype=np.int8)
        handstr = cpp_poker.get_hand_rank_52_plo(hand_2d=h, board_2d=b)
        assert handstr == 1240771

    def test_get_hand_rank_all_hands_on_given_boards(self):
        board_1d = np.array([12,14,25,31,50], dtype=np.int8)
        handranks = self._env.get_hand_rank_all_hands_on_given_boards(
            boards_1d=board_1d.reshape(1, board_1d.shape[0]), lut_holder=self._env.lut_holder)[0]
        assert np.sum(handranks) == 687775072


if __name__ == '__main__':
    unittest.main()
