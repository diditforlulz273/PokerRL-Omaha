# Copyright (c) 2019 Eric Steinberger


"""
This file is not runable; it's is a template to show how you could play against your algorithms. To do so,
replace "YourAlgorithmsEvalAgentCls" with the EvalAgent subclass (not instance) of your algorithm.

Note that you can see the AI's cards on the screen since this is just a research application and not meant for actual
competition. The AI can, of course, NOT see your cards.
"""
import time
from os.path import dirname, abspath

import numpy as np

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.AgentTournament import AgentTournament

if __name__ == '__main__':
    #path_to_first_eval_agent = dirname(abspath(__file__)) + "/../trained_agents/NLH_44steps_SINGLE.pkl"
    path_to_second_eval_agent = dirname(abspath(__file__)) + "/../trained_agents/eval_agentSINGLE_6threads32steps.pkl"
    path_to_first_eval_agent = dirname(abspath(__file__)) + "/../trained_agents/NLH_31steps_old_SINGLE.pkl"
    #path_to_second_eval_agent = dirname(abspath(__file__)) + "/../trained_agents/NLH_44steps_SINGLE.pkl"

    eval_agent_first = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_first_eval_agent)
    eval_agent_second = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_second_eval_agent)
    # assert eval_agent_first.t_prof.name == eval_agent_second.t_prof.name

    game = AgentTournament(env_cls=eval_agent_first.env_bldr.env_cls,
                           env_args=eval_agent_first.env_bldr.env_args,
                           eval_agent_1=eval_agent_first,
                           eval_agent_2=eval_agent_second,
                           )

    game.run(n_games_per_seat=100000)
