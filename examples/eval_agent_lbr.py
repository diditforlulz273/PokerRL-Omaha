# Copyright (c) 2019 Eric Steinberger


"""
Evaluation module.
Runs series of LBR rollouts to calculate approximate exploitability vs.
worst case opponent.
Can be used for overall agent exploitability evaluation and thus indirect comparison
or model optimisation.

A path to pickled agent should be set in path_to_eval_agent
"""
import os
from os.path import dirname, abspath
from os.path import join as ospj

from PokerRL.game import bet_sets
from PokerRL.game.Poker import Poker

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase
from PokerRL.rl.MaybeRay import MaybeRay

if __name__ == '__main__':
    path_to_eval_agent = dirname(abspath(__file__)) + "/../trained_agents/deepl-st24.pkl"
    eval_agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_eval_agent)

    if eval_agent.t_prof.DISTRIBUTED:
        from DeepCFR.workers.chief.dist import Chief
    else:
        from DeepCFR.workers.chief.local import Chief

    # Here we correct path to current local machine settings,
    # cuz in case model trained on another machine, path might be infeasible
    # which will cause a error in DriverBase while pickling TrainingProfile to disk
    # or initializing Crayon server
    def get_root_path():
        return "C:\\" if os.name == 'nt' else os.path.expanduser('~/')
    data_path = ospj(get_root_path(), "poker_ai_data")
    eval_agent.t_prof.path_trainingprofiles = ospj(data_path, "TrainingProfiles")
    eval_agent.t_prof.path_log_storage = ospj(data_path, "logs")

    # Set False for debug purposes - with True Ray Actors are untraceble
    # Doesent work now, should be equal to the origin
    # eval_agent.t_prof.DISTRIBUTED = False

    # Overwrite parameters of LBR Evaluator (this evaluator type should be used
    # in the original EvalAgent TrainingProfile!)
    eval_agent.t_prof.module_args['lbr'].DISTRIBUTED = True
    eval_agent.t_prof.module_args['lbr'].n_workers = 4
    eval_agent.t_prof.module_args['lbr'].use_gpu_for_batch_eval = False
    eval_agent.t_prof.module_args['lbr'].n_lbr_hands = 2000
    eval_agent.t_prof.module_args['lbr'].lbr_check_to_round = Poker.TURN
    eval_agent.t_prof.module_args['lbr'].lbr_bet_set = eval_agent.env_bldr.env_args.bet_sizes_list_as_frac_of_pot


    ctrl = DriverBase(t_prof=eval_agent.t_prof,
                      eval_methods={"lbr": 1,},
                      chief_cls=Chief,
                      eval_agent_cls=EvalAgentDeepCFR,
                      n_iterations=None,
                      iteration_to_import=None,
                      name_to_import=None
                      )

    ray = MaybeRay(runs_distributed=eval_agent.t_prof.DISTRIBUTED, runs_cluster=eval_agent.t_prof.CLUSTER)

    # get strategy buffers from eval_agents
    std = eval_agent._state_dict()
    strategy_buffers = std['strategy_buffers']

    # push strategies to Chief
    ray.remote(ctrl.chief_handle.load_checkpoint_param,std)

    # Set net indexes to get for evaluation
    # Quite logical to use the latest nets, so
    num_nets = []
    for s in strategy_buffers:
        num_nets.append(len(s['nets']) - 1)

    # Also possible to uncomment and evaluate all strategies weighted in the SD-CFR way,
    # but its Nstrategies times slower!
    # num_nets = (0,0)

    # Finally start evaluation
    ctrl.evaluate(num_nets)

    # In case of distributed run we have to make current python thread wait till all
    # Ray LBR workers finish rollouts to see evaluation result in console, so we wait for a keypress to close
    input(f"\nComputing, type in any character when done to calculate the result \n\n")

    new_v, experiment_names = ray.get(ray.remote(ctrl.chief_handle.get_new_values))
    conf_upper = new_v[experiment_names[2]]['Evaluation/MBB_per_G'][0][1]
    conf_lower = new_v[experiment_names[1]]['Evaluation/MBB_per_G'][0][1]
    lbr_res = new_v[experiment_names[3]]['Evaluation/MBB_per_G'][0][1]
    print(f"Played hands for each player: {eval_agent.t_prof.module_args['lbr'].n_lbr_hands}\n")
    print(f"Upper Confidence 95% interval milliBB/hand: {conf_upper}\n")
    print(f"LBR Exploitability milliBB/hand: {lbr_res}\n")
    print(f"Lower Confidence 95% interval milliBB/hand: {conf_lower}\n")
