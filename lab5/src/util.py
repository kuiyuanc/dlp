import conv_dqn
import dqn
import enhanced_conv_dqn


def config(*, task: int) -> tuple[str, str, bool, type, type]:
    if task == 1:
        env_name = "CartPole-v1"
        enhance = "vanilla"
        atari = False
        DQN = dqn.DQN
        Agent = dqn.DQNAgent
    elif task == 2:
        env_name = "Pong-v5"
        enhance = "vanilla"
        atari = True
        DQN = conv_dqn.ConvDQN
        Agent = conv_dqn.ConvDQNAgent
    elif task == 3:
        env_name = "Pong-v5"
        enhance = "enhanced"
        atari = True
        DQN = conv_dqn.ConvDQN
        Agent = enhanced_conv_dqn.RainbowConvDQNAgent
    else:
        raise ValueError("Invalid task")

    return env_name, enhance, atari, DQN, Agent
