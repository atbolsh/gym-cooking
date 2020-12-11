# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple

import gym


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")

    return parser.parse_args()


def fake_parse_arguments(level, num_agents,\
                         max_num_timesteps = 100,\
                         max_num_subtasks = 14,\
                         seed = 1,\
                         with_image_obs=False,\
                         beta = 1.3,\
                         alpha = 0.01,\
                         tau = 2,\
                         cap = 75,\
                         main_cap = 1000,\
                         play = False,\
                         record = False,\
                         model1 = None,\
                         model2 = None,\
                         model3 = None,\
                         model4 = None):
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")
    nm = parser.parse_args()
    nm.level = level
    nm.num_agents = num_agents
    nm.max_num_timesteps = max_num_timesteps
    nm.max_num_subtasks = max_num_subtasks
    nm.seed = seed
    nm.with_image_obs = with_image_obs
    nm.beta = beta
    nm.alpha = alpha
    nm.tau = tau
    nm.cap = cap
    nm.main_cap = main_cap
    nm.play = play
    nm.record = record
    nm.model1 = model1
    nm.model2 = model2
    nm.model3 = model3
    nm.model4 = model4
    return nm


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def initialize_agents(arglist):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    real_agent = RealAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                    real_agents.append(real_agent)

    return real_agents

def main_loop(arglist):
    """The main loop for running experiments."""
    init_obj = init_main_loop(arglist)
    env = init_obj[0]
    while not env.done():
        init_obj = main_loop_step(init_obj)
        env = init_obj[0]
    # Saving final information before saving pkl file
    return finish_main_loop(init_obj)

def init_main_loop(arglist):
    model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
    assert len(list(filter(lambda x: x is not None,
        model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
    fix_seed(seed=arglist.seed)

    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    # game = GameVisualize(env)
    real_agents = initialize_agents(arglist=arglist)

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)
    return env, obs, real_agents, bag

def main_loop_step(init_obj): # init_obj is (env, obs, real_agents, bag)
    env, obs, real_agents, bag = init_obj
    if env.done():
        print("Environment already done; can't take step.")
        return init_obj

    action_dict = {}

    for agent in real_agents:
        action = agent.select_action(obs=obs)
        action_dict[agent.name] = action

    obs, reward, done, info = env.step(action_dict=action_dict)
    print(type(obs))

    # Agents
    for agent in real_agents:
        agent.refresh_subtasks(world=env.world)

    # Saving info
    bag.add_status(cur_time=info['t'], real_agents=real_agents)
    return env, obs, real_agents, bag

def finish_main_loop(init_obj):
    # Saving final information before saving pkl file
    env, obs, real_aents, bag = init_obj

    bag.set_collisions(collisions=env.collisions)
    bag.set_termination(termination_info=env.termination_info,
            successful=env.successful)
    return env, obs, real_agents, bag

# if __name__ == '__main__':
#     arglist = parse_arguments()
#     if arglist.play:
#         env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
#         env.reset()
#         game = GamePlay(env.filename, env.world, env.sim_agents)
#         game.on_execute()
#     else:
#         main_loop(arglist=arglist)
# 
# 
