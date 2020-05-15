from PokerRL.game.games import DiscretizedNLHoldem, PLO
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from PokerRL.game import bet_sets
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="NLH_EXPLOITABILITY_PLO",
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
                                         n_actions_traverser_samples=4, # 3 is the default, 4 is current max for b_2
                                         #number of traversals gives some amount of otcomes to train network on
                                         #mult = 1...4, buffer appends every() step with new data
                                         n_traversals_per_iter=30,
                                         #number of mini_batch fetches and model updates on each step
                                         n_batches_adv_training=801, #1024
                                         n_batches_avrg_training=2048, #2048

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,

                                         #amount of batch to feed to NN at once, fetched from buffer randomly.
                                         mini_batch_size_adv=512, #256
                                         mini_batch_size_avrg=512, #512
                                         init_adv_model="random",  # warm start neural weights with init from last iter
                                         init_avrg_model="random",
                                         #use_pre_layers_avrg=False,  # shallower nets

                                         lr_avrg=0.001,
                                         game_cls=PLO, #PLO or DiscretizedNLHoldem
                                         env_bldr_cls=VanillaEnvBuilder,
                                         agent_bet_set=bet_sets.PL_2,
                                         n_seats=2,
                                         start_chips=10000,

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                             #EvalAgentDeepCFR.EVAL_MODE_AVRG_NET
                                         ),

                                         # enables simplified obs. Default works also for 3+ players
                                         use_simplified_headsup_obs=True,

                                         log_verbose=True,
                                         lbr_args=LBRArgs(lbr_bet_set=bet_sets.PL_2,
                                                         n_lbr_hands_per_seat=100,
                                                         lbr_check_to_round=Poker.TURN,  # recommended to set to Poker.TURN for 4-round games.
                                                         n_parallel_lbr_workers=1,
                                                         use_gpu_for_batch_eval=False,
                                                         DISTRIBUTED=True,
                                         ),
                                         ),
                  eval_methods={
                      "lbr": 4,
                  },
                  n_iterations=30)
    ctrl.run()