from SL.World import World
import SL.TensorboardLogger as TensorboardLogger
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
    if 'Stacking' == environment_name:
        import Tasks.Stacking as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingOne' == environment_name:
        import Tasks.Stacking_one as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingBinary' == environment_name:
        import Tasks.Stacking_binary as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingBinaryZ' == environment_name:
        import Tasks.Stacking_binary_z as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingBinaryZcopy' == environment_name:
        import Tasks.Stacking_binary_z_copy as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingBinaryZcopyStack' == environment_name:
        import Tasks.Stacking_binary_z_copy_stack as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingBinaryXYZcopyStack' == environment_name:
        import Tasks.Stacking_binary_xyz_copy_stack as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingBinaryZcopyStackNoise' == environment_name:
        import Tasks.Stacking_binary_z_copy_stack_noise as Stacking
        env_fun = Stacking.Stacking
    elif 'StackingBinaryXYZSum' == environment_name:
        import Tasks.Stacking_binary_xyz_sum as Stacking
        env_fun = Stacking.Stacking
    else:
        raise ValueError('Environment name not set or unknown. Current value: {}'.format(environment_name))

    agent_fun = None
    if agent_name == 'SampleAgent':
        import Agents.SampleAgent as SampleAgent
        agent_fun = SampleAgent.SampleAgent
    elif agent_name == 'DnhcAgent':
        import Agents.DnhcAgent as DnhcAgent
        agent_fun = DnhcAgent.DnhcAgent
    elif agent_name == 'DnhcALUAgent':
        import Agents.DnhcALUAgent as DnhcAgent
        agent_fun = DnhcAgent.DnhcAgent
    elif agent_name == 'DnhcALUAgent_wM':
        import Agents.DnhcALUAgent_wM as DnhcAgent
        agent_fun = DnhcAgent.DnhcAgent
    elif agent_name == 'DncAgent':
        import Agents.DncAgent as DncAgent
        agent_fun = DncAgent.DncAgent
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
    if config['execution']:
        world.execute_step()
    else:
        world.execute(max_episodes)

if __name__ == "__main__":
    main()
