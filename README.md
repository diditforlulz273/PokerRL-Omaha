# PokerRL Omaha

A fork of the original [Framework for Multi-Agent Deep Reinforcement Learning in Poker games](https://github.com/EricSteinberger/PokerRL) by [Eric Steinberger](https://github.com/EricSteinberger)
Combined with [SD-CFR](https://github.com/EricSteinberger/Deep-CFR)

Now works with Omaha Poker.

Internet lacks any open-source Omaha Poker Reinforcement Learning code, so I created this part myself.

It was hard to stop digging in the original code masterpiece so there are some additional functionality and improvements.


## Changes
Only differences are noticed here, compilation and basic usage are exhaustively explained in the original repos.
Some new dependencies exist, full dep. list is in requirements_dist.txt

#### Fully functional Pot Limit Omaha game:

 - Works for 2-6 players.
 - Smoothly integrated in code, so functionality of the original PokerRL is preserved.
 - All the look up tables are rewritten in pure Python, although generation of
  some of them is not fully vectorized, so takes up to 10 secs to build on the start.
 - Uses the original hand evaluator with Omaha combinations on top. Being naive and slow, it slightly impacts the speed of LBR rollouts. 

Use game type 'PLO' to start, an example is provided in 'examples/PLO_training_start.py'.

#### Preflop Hand Bucketing 
 - Works for Hold'em and Omaha.
 - Improves Neural Network convergence at the beginning of the training, thus decreases overall convergence time.
 - Buckets together all preflop-isomorphic hands, e.g. AsKh and AdKc - suits doesen't matter without flop.
 - Uses additional bucketed look up table with empty suit bits. 
 - Found in neural network modules FLAT, FLAT2 and CNN.
 
 Could be named a handcrafted feature which slightly conflicts
  with a general self-play education idea, but PokerRL actually buckets isomorphic flop cards, and everyone does it as well.
  
#### Optimized Dense Residual Neural Network 
 - MainPokerModuleFLAT2 which is used by setting nn_type='dense_residual'.
 - NN which is 2x deeper but has roughly the same
 computational complexity as the original FLAT.
 - Yields around 11% faster training in terms of loss decrease.
 - Significantly outperforms the original NN agent in PLO on any training step tested in h2h.
 
#### Convolutional Neural Network 
 - Which hasn't explored much and doesen't work well ATM. The idea is to
  explore network potency to pick all the parameters without human segmentation from 2D array. First 2-4 rows are private cards,
  next 5 are board cards and the last one is a vector of stacks and bets happened before.
  Total array size is 8X24.
 
#### Leaky ReLU usage for all NNs
 - negative slope of 0.1 is tested to improve loss decrease speed by 2-6% at no cost
 
#### Standalone Head to Head Agent evaluator
 - Standalone module written with takeaways from the original h2h evaluator of PokerRL
 - Is handy to evaluate different agents against each other.
 Can be found in 'examples/interactive_agent_vs_agent.py', a short parameter description is inside.
 Class AgentTournament is extended to hold the functionality.

#### Standalone LBR Agent evaluator
 - Standalone module written with takeaways from the original h2h evaluator of PokerRL
 - Is handy to evaluate an agent with LBR method.
 Can be found in 'examples/eval_agent_lbr.py', a short parameter description is inside.
 
#### Hand Logger for H2H Evaluator
 - Writes actual hands played in close-to-PokerStars format in .txt file.
 - Enabled by default in Standalone H2H Evaluator.
 - Modifies classes PokerEnv and AgentTournament and to catch all the activity.
 - Introduces HandHistoryLogger class.
 Allows manual hand history reading and storing in plain text, could also be
  loaded in PT4 for basic analysis, although not fully mimics the correct
  HH format - the only goal was to make played games easy readable.
  
#### Slightly altered Traversal Data generation scheme
 - now n_traversals_per_iter sets the exact number of data entries created for each player
 (was a number of external traverser rollouts before, which has been giving quite unstable amounts)
 
#### Bug fixes
 - I don't remember them all, but among most important are use of deprecated torch tensor classes
 which crashed the GPU code on recent torch versions, some index miscalculations and wrong unsqueezes.
  
  