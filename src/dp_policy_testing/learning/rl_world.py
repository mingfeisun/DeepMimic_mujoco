import numpy as np
import learning.agent_builder as AgentBuilder
import learning.tf_util as TFUtil
from learning.rl_agent import RLAgent
from util.logger import Logger

class RLWorld(object):
    def __init__(self, arg_parser):
        TFUtil.disable_gpu()

        self.arg_parser = arg_parser
        self._enable_training = True
        self.train_agents = []
        self.parse_args(arg_parser)

        self.build_agents()
        
        return

    def parse_args(self, arg_parser):
        self.train_agents = self.arg_parser.parse_bools('train_agents')
        num_agents = 1
        assert(len(self.train_agents) == num_agents or len(self.train_agents) == 0)

        return

    def build_agents(self):
        num_agents = 1
        self.agents = []

        Logger.print('')
        Logger.print('Num Agents: {:d}'.format(num_agents))

        agent_files = self.arg_parser.parse_strings('agent_files')
        assert(len(agent_files) == num_agents or len(agent_files) == 0)

        model_files = self.arg_parser.parse_strings('model_files')
        assert(len(model_files) == num_agents or len(model_files) == 0)

        output_path = self.arg_parser.parse_string('output_path')
        int_output_path = self.arg_parser.parse_string('int_output_path')

        for i in range(num_agents):
            curr_file = agent_files[i]
            curr_agent = self._build_agent(i, curr_file)

            if curr_agent is not None:
                curr_agent.output_dir = output_path
                curr_agent.int_output_dir = int_output_path
                Logger.print(str(curr_agent))

                if (len(model_files) > 0):
                    curr_model_file = model_files[i]
                    if curr_model_file != 'none':
                        curr_agent.load_model(curr_model_file)

            self.agents.append(curr_agent)
            Logger.print('')
        return

    def get_action(self, s, g):
        return self.agents[0].get_action(s, g)

    def _build_agent(self, id, agent_file):
        Logger.print('Agent {:d}: {}'.format(id, agent_file))
        if (agent_file == 'none'):
            agent = None
        else:
            agent = AgentBuilder.build_agent(self, id, agent_file)
            assert (agent != None), 'Failed to build agent {:d} from: {}'.format(id, agent_file)
        
        return agent
        