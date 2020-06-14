
"""
HU play module for agents comparison.
Takes two pickled EvalAgent classes as inputs,
number of games to play - for each agent, so the total amount is N*2,
and a path to file for a hand history logger, which saves play history in PokerStars format,
so it could be analyzed with PokerTracker4 later.

Returns avg. winnings of agents in milliBB/hand in the end.

As for now HH files tested and mostly work correctly only with PT4 software.
"""
from os.path import dirname, abspath

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.AgentTournament_hu import AgentTournament

if __name__ == '__main__':

    path_to_first_eval_agent = dirname(abspath(__file__)) + "/../trained_agents/17_dense.pkl"
    path_to_second_eval_agent = dirname(abspath(__file__)) + "/../trained_agents/19_bucket_last_2xspeed.pkl"

    eval_agent_first = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_first_eval_agent)
    eval_agent_second = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_second_eval_agent)
    # assert eval_agent_first.t_prof.name == eval_agent_second.t_prof.name

    game = AgentTournament(env_cls=eval_agent_first.env_bldr.env_cls,
                           env_args=eval_agent_first.env_bldr.env_args,
                           eval_agent_1=eval_agent_first,
                           eval_agent_2=eval_agent_second,
                           logfile="../HandHistory/AgentTourney.txt"   # "../HandHistory/AgentTourney.txt" or None
                           )

    game.run(n_games_per_seat=10000)
