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

# in case we need reproductability
#"""
seed = 105
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
#"""

if __name__ == '__main__':
    ctrl = Driver(iteration_to_import=30, name_to_import='NLH_1.5m_10mX2-b2048-last-patience200-Leaky-lr0.004_',
                 t_prof=TrainingProfile(name="CNN_test_on_checkpoint309_feedforward_batch10k",
                                        nn_type="convolutional",

                                        DISTRIBUTED=False,
                                        CLUSTER=False,
                                        n_learner_actor_workers=1,  # 20 workers
                                        device_training='cuda',
                                        device_inference='cuda',
                                        device_parameter_server='cuda',

                                        max_buffer_size_adv=1500000,  # 1.5e6
                                        export_each_net=False,
                                        # path_strategy_nets="",
                                        checkpoint_freq=5,  # produces A SHITLOAD of Gbs!
                                        eval_agent_export_freq=1,  # produces GBs!

                                        # How many actions out of all legal on current step to branch randomly = action bredth limit
                                        n_actions_traverser_samples=4,
                                        # 3 is the default, 4 is the current max for b_2
                                        # number of traversals equals to the number of new data entries in adv_buf
                                        n_traversals_per_iter=1500,
                                        # number of mini_batch fetches and model updates on each step
                                        n_batches_adv_training=1000,  # 5000

                                        use_pre_layers_adv=True,
                                        n_cards_state_units_adv=192,
                                        n_merge_and_table_layer_units_adv=64,  # 64
                                        n_units_final_adv=64,  # 64
                                        dropout_adv=0.0,
                                        lr_patience_adv=1000,  # decrease by a factor 0.5(in PSWorker)
                                        lr_adv=0.004,  # if no better after 150 batches

                                        # amount of batch to feed to NN at once, fetched from buffer randomly.
                                        mini_batch_size_adv=2000,  # 512
                                        init_adv_model="random",  # last, random

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
                  n_iterations=33)

    ctrl.run()


