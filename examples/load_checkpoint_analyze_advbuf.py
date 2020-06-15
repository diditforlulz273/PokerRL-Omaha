import numpy as np
import random
import torch

from PokerRL.game.games import DiscretizedNLHoldem, PLO, Flop5Holdem
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from PokerRL.game import bet_sets
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver


def hand_and_round_eq(hand, round, ctrl):
    lut_holder, state_dicts = ctrl.la_handles[0].get_lut_and_statedicts()
    rng = lut_holder.get_range_idx_from_hole_cards(hand)
    buf_adv = state_dicts[player]['adv'].numpy()
    buf_pubobs = state_dicts[player]['base']['pub_obs_buffer'].numpy()
    buf_range = state_dicts[player]['base']['range_idx_buffer'].numpy()

    # select all indexes where hand = ours
    hand_idxs = np.where(buf_range == rng)[0]

    # select all public observations where game round = round. rounds are marked by 1
    # starting from 14 to 17 corresponding to preflop..river
    selected_pubobs_idxs = np.where(buf_pubobs[:, 14+round] == 1)[0]

    # now we find an intersection between indexes of two conditions
    intersect_idxs = np.intersect1d(hand_idxs, selected_pubobs_idxs)
    adv_result = buf_adv[intersect_idxs]
    return np.mean(adv_result, 0), len(adv_result)


if __name__ == '__main__':
    ctrl = Driver(iteration_to_import=30, name_to_import='NLH_1.5m_10mX2-b2048-last-patience200-Leaky-lr0.004_',
                 t_prof=TrainingProfile(name="NLH_1.5m_10mX2-b2048-last-patience200-Leaky-lr0.004",
                                        nn_type="feedforward",

                                        DISTRIBUTED=False,
                                        CLUSTER=False,
                                        n_learner_actor_workers=1,  # 20 workers

                                        max_buffer_size_adv=1500000,  # 1.5e6
                                        export_each_net=False,
                                        # path_strategy_nets="",
                                        checkpoint_freq=5,  # produces A SHITLOAD of Gbs!
                                        eval_agent_export_freq=1,  # produces GBs!

                                        # How many actions out of all legal on current step to branch randomly = action bredth limit
                                        n_actions_traverser_samples=4,
                                        # 3 is the default, 4 is the current max for b_2
                                        # number of traversals gives some amount of otcomes to train network on
                                        # mult = 1...4, buffer appends every() step with new data
                                        n_traversals_per_iter=3500,
                                        # number of mini_batch fetches and model updates on each step
                                        n_batches_adv_training=6000,  # 5000

                                        use_pre_layers_adv=True,
                                        n_cards_state_units_adv=192,
                                        n_merge_and_table_layer_units_adv=64,  # 64
                                        n_units_final_adv=64,  # 64
                                        dropout_adv=0.0,
                                        lr_patience_adv=750,  # decrease by a factor 0.5(in PSWorker)
                                        lr_adv=0.004,  # if no better after 150 batches

                                        # amount of batch to feed to NN at once, fetched from buffer randomly.
                                        mini_batch_size_adv=10000,  # 512
                                        init_adv_model="last",  # last, random

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
                                                         n_lbr_hands_per_seat=1,
                                                         lbr_check_to_round=Poker.TURN,
                                                         # recommended to set to Poker.TURN for 4-round games.
                                                         n_parallel_lbr_workers=1,
                                                         use_gpu_for_batch_eval=False,
                                                         DISTRIBUTED=False,
                                                         ),
                                        ),
                  eval_methods={
                      "lbr": 99,  # lbr, br, h2h
                  },
                  n_iterations=64)

    # here we find all particular aces and their avg regrets grouped by actions where round=preflop
    hand = np.array([[12, 2], [12, 0]])
    player = 0
    g_round = Poker.TURN
    eq = hand_and_round_eq(hand=hand, round=g_round, ctrl=ctrl)
    print(f'regrets: {eq[0]} \ncases: {eq[1]}')


