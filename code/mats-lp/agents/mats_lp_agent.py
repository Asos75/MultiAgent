from mcts_cpp.cppmcts import MCTSConfig, MCTSInference


class MATSLPAgent:
    def __init__(self, num_expansions=250, num_threads=4, pb_c_init=4.44):
        self.algo = MCTSInference(MCTSConfig(
            num_expansions=num_expansions,
            num_threads=num_threads,
            pb_c_init=pb_c_init,
        ))

    def act(self, obs):
        return self.algo.act(obs)