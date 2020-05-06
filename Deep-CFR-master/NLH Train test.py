from PokerRL.game.games import DiscretizedNLHoldem
from PokerRL.eval.head_to_head.H2HArgs import H2HArgs

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="NLH_EXPLOITABILITY",
                                         nn_type="feedforward",

                                         DISTRIBUTED=False,
                                         CLUSTER=False,
                                         n_learner_actor_workers=3,  # 20 workers

                                         max_buffer_size_adv=1e6,
                                         max_buffer_size_avrg=1e6,
                                         export_each_net=False,
                                         checkpoint_freq=99999999,
                                         eval_agent_export_freq=1,  # produces GBs!

                                         n_traversals_per_iter=2048,
                                         n_batches_adv_training=2048,
                                         n_batches_avrg_training=2048,

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,

                                         mini_batch_size_adv=2048,
                                         mini_batch_size_avrg=2048,
                                         init_adv_model="last",  # warm start neural weights with init from last iter
                                         init_avrg_model="random",
                                         #use_pre_layers_avrg=False,  # shallower nets

                                         lr_avrg=0.001,
                                         game_cls=DiscretizedNLHoldem,

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                             EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,  # Deep CFR
                                         ),

                                         # enables simplified obs. Default works also for 3+ players
                                         use_simplified_headsup_obs=True,

                                         log_verbose=True,
                                         h2h_args=H2HArgs(
                                             n_hands=5000,  # this is per seat; so in total 10k hands per eval
                                         ),
                                         ),
                  eval_methods={
                      "h2h": 6,
                  },
                  n_iterations=25)
    ctrl.run()
