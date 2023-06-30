import logging

# from gym.envs import register
from gym.envs.registration import registry, register


LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(
        id='BrokenAnt-v3',
        entry_point='examples.envs.custom_envs:BrokenAntEnv',
        max_episode_steps=1000,
        reward_threshold=6000.0,
        kwargs={"terminate_when_unhealthy": False}
    )

    register(
        id='BrokenHalfCheetah-v3',
        entry_point='examples.envs.custom_envs:BrokenHalfCheetahEnv',
        max_episode_steps=1000,
        reward_threshold=4800.0,
    )

    register(
        id='BrokenHumanoid-v3',
        entry_point='examples.envs.custom_envs:BrokenHumanoidEnv',
        max_episode_steps=1000,
        kwargs={'broken_joint': 0}
    )

    register(
        id='Reacher-v0',
        entry_point='examples.envs.reacher_7dof:Reacher7DofEnv',
        max_episode_steps=100,
    )

    register(
        id='BrokenReacher-v0',
        entry_point='examples.envs.custom_envs:BrokenReacherEnv',
        max_episode_steps=100,
    )

    register(
        id='HalfCheetahObstacle-v3',
        entry_point='examples.envs.custom_envs:HalfCheetahDirectionEnv',
        max_episode_steps=1000,
    )

    register(
        id='CustomAnt-v0',
        entry_point='examples.envs.custom_envs:CustomAntEnv',
        max_episode_steps=1000,
        kwargs={'gear': 30, 'disabled': 0}
    )

    register(
        id='DisabledAnt-v2',
        entry_point='examples.envs.custom_envs:CustomAntEnv',
        max_episode_steps=1000,
        kwargs={'gear': 30, 'disabled': 2}
    )

    register(
        id='DisabledAnt-v1',
        entry_point='examples.envs.custom_envs:CustomAntEnv',
        max_episode_steps=1000,
        kwargs={'gear': 30, 'disabled': 1}
    )

    register(
        id='UMaze-v0',
        entry_point='examples.envs.maze_envs:MazeEnd_PointMass',
        max_episode_steps=1000,
        kwargs={'maze_id': 0}
    )

    register(
        id='InverseUMaze-v0',
        entry_point='examples.envs.maze_envs:MazeEnd_PointMass',
        max_episode_steps=1000,
        kwargs={'maze_id': 4}
    )

    register(
        id='InverseUMaze-v1',
        entry_point='examples.envs.maze_envs:MazeEnd_PointMass',
        max_episode_steps=1000,
        kwargs={'maze_id': 5}
    )

    