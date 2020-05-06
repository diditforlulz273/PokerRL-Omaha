# Copyright (c) 2019 Eric Steinberger


"""
This script runs 150 iterations of Linear CFR in an HLH poker game with actions {DEFINE}.
It will store logs and tree files on your C: drive.  - NOPE
"""

from PokerRL.cfr.LinearCFR import LinearCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLHoldem
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

if __name__ == '__main__':
    from PokerRL._.CrayonWrapper import CrayonWrapper

    n_iterations = 20
    name = "LinCFR_NLH"

    # Passing None for t_prof will is enough for ChiefBase. We only use it to log; This CFR impl is not distributed.
    chief = ChiefBase(t_prof=None)
    crayon = CrayonWrapper(name=name,
                           path_log_storage=None,
                           chief_handle=chief,
                           runs_distributed=False,
                           runs_cluster=False)
    cfr = LinearCFR(name=name,
                    game_cls=DiscretizedNLHoldem,
                    agent_bet_set=bet_sets.PL_3,
                    chief_handle=chief)

    for iter_id in range(n_iterations):
        print("Iteration: ", iter_id)
        cfr.iteration()
        crayon.update_from_log_buffer()
        crayon.export_all(iter_nr=iter_id)
