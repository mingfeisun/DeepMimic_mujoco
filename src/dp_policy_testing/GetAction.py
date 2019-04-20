import numpy as np
import sys
import random

from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.util as Util

def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if (arg_file != ''):
        succ = arg_parser.load_file(arg_file)
        assert succ, Logger.print('Failed to load args from: ' + arg_file)

    rand_seed_key = 'rand_seed'
    if (arg_parser.has_key(rand_seed_key)):
        rand_seed = arg_parser.parse_int(rand_seed_key)
        rand_seed += 1000 * MPIUtil.get_proc_rank()
        Util.set_global_seeds(rand_seed)

    return arg_parser

class ActionGiver():
    def __init__(self, args):
        arg_parser = build_arg_parser(args)
        self.world = RLWorld(arg_parser)

    def get_ac(self, s, g):
        return self.world.get_action(s, g)

if __name__ == '__main__':
    # args = sys.argv[1:]
    args = ['--arg_file', 'args/run_humanoid3d_walk_args.txt']
    action_giver = ActionGiver(args)
    state = np.zeros(197)
    goal = np.zeros(1)
    action = action_giver.get_ac(state, goal)
    print(action)
    print('Shape:', action.shape)