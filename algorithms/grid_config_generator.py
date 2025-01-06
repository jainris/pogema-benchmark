from pogema import GridConfig

def add_grid_config_args(parser):
    parser.add_argument("--map_type", type=str, default="RandomGrid")
    parser.add_argument("--map_h", type=int, default=20)
    parser.add_argument("--map_w", type=int, default=20)
    parser.add_argument("--robot_density", type=float, default=0.025)
    parser.add_argument("--obstacle_density", type=float, default=0.1)
    parser.add_argument("--max_episode_steps", type=int, default=128)
    parser.add_argument("--obs_radius", type=int, default=4)
    parser.add_argument("--collision_system", type=str, default="soft")
    parser.add_argument("--on_target", type=str, default="nothing")

    return parser


def generate_grid_config_from_env(env):
    config = env.grid.config
    return GridConfig(
        num_agents=config.num_agents,  # number of agents
        size=config.size,  # size of the grid
        density=config.density,  # obstacle density
        seed=config.seed,
        max_episode_steps=config.max_episode_steps,  # horizon
        obs_radius=config.obs_radius,  # defines field of view
        observation_type=config.observation_type,
        collision_system=config.collision_system,
        on_target=config.on_target,
        map=env.grid.get_obstacles(ignore_borders=True).tolist(),
        agents_xy=env.grid.get_agents_xy(ignore_borders=True),
        targets_xy=env.grid.get_targets_xy(ignore_borders=True),
    )


def grid_config_generator_factory(
    map_type,
    map_w,
    map_h,
    num_agents,
    obstacle_density,
    obs_radius,
    collision_system,
    on_target,
    max_episode_steps=None,
):
    if map_type == "RandomGrid":
        assert map_h == map_w, (
            f"Expect height and width of random grid to be the same, "
            f"but got height {map_h} and width {map_w}"
        )

        def _grid_config_generator(seed):
            return GridConfig(
                num_agents=num_agents,  # number of agents
                size=map_w,  # size of the grid
                density=obstacle_density,  # obstacle density
                seed=seed,  # set to None for random
                # obstacles, agents and targets
                # positions at each reset
                max_episode_steps=max_episode_steps,  # horizon
                obs_radius=obs_radius,  # defines field of view
                observation_type="MAPF",
                collision_system=collision_system,
                on_target=on_target,
            )
    else:
        raise ValueError(f"Unsupported map type: {map_type}.")
    return _grid_config_generator
