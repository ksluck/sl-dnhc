from SL.World import World
import SL.TensorboardLogger as TensorboardLogger
import Agents.SampleAgent as SampleAgent
import json
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file', default='configs/config.json', type=str, help='Configuration to load.')
    args = parser.parse_args()
    config = json.load(open(args.config_file))

    environment_name = config['environment']
    agent_name = config['agent']
    logger_name = config['logger']
    render_first_env = config['rendering']
    max_episodes = config['episodes']
    exp_id = config['exp_id']
    config['parent_dir'] = config['filepath'] + '/' + str(exp_id)
    config['checkpoint_dir'] = config['checkpoint_filepath'] + '/' + str(exp_id) + config['checkpoint_dir']

    env_fun = None
    if 'Acrobot' == environment_name:
        env_fun = Acrobot
    else:
        raise ValueError('Environment name not set or unknown. Current value: {}'.format(environment_name))

    agent_fun = None
    if agent_name == 'SampleAgent':
        agent_fun = SampleAgent.SampleAgent
    else:
        raise ValueError('Agent name not set or unknown. Current Value: {}'.format(agent_name))
    print('Agent: {}'.format(agent_name))

    if logger_name == 'tensorboard':
        logger_fun = TensorboardLogger.TensorboardLogger
    else:
        raise ValueError('Logger name not set or unknown. Current Value: {}'.format(logger_name))

    print('Logger: {}'.format(logger_name))

    config_agent = config[agent_name]
    config_agent['parent_dir'] = config['parent_dir']
    config_logger = config[logger_name]
    config_logger['parent_dir'] = config['parent_dir']
    config_env = config[environment_name]
    config_env['parent_dir'] = config['parent_dir']

    envs = env_fun(config_env, render_first_env)
    agents = agent_fun(envs, config=config_agent)
    #agents = [TeeDynasaur(envs[0], config=config_teedyna), TeeDynasaur(envs[1], config=config_teedyna), TeeDynasaur(envs[2], config=config_teedyna)]
    logger = logger_fun(config_logger)
    world = World(agents, envs, logger, 0, config)
    world.execute(max_episodes)

if __name__ == "__main__":
    main()
