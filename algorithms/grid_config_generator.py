import numpy as np

from pogema import GridConfig
from pogema.generator import bfs


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

    parser.add_argument("--min_dist", type=int, default=None)

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


MOVES = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
START_ID = 2


def _get_components(obstacles, map_w, start_id=START_ID, moves=MOVES):
    grid = obstacles.copy()

    components = bfs(grid, moves, map_w, start_id, free_cell=0)
    return grid, components


def generate_start_target_pairs(obstacles, map_w, min_dist):
    # All possible pairs [start_x, start_y, target_x, target_y]
    possible_pairs = np.ones((map_w, map_w, map_w, map_w), dtype=int)

    # Removing obstacles
    possible_pairs *= 1 - obstacles[:, :, None, None]
    possible_pairs *= 1 - obstacles[None, None, :, :]

    # Removing pairs that are too close
    locs = np.zeros((map_w, map_w, 2), dtype=int)
    locs[:, :, 0] = np.arange(map_w)[:, None]
    locs[:, :, 1] = np.arange(map_w)[None, :]

    pdist = np.sum(
        np.abs(locs[:, :, None, None] - locs[None, None, :, :]), axis=-1
    )  # Manhattan Distance

    possible_pairs *= pdist >= min_dist

    # Removing unreachable pairs
    component_map, components = _get_components(obstacles, map_w, start_id=START_ID)

    reachable_pairs = np.zeros((map_w, map_w, map_w, map_w), dtype=bool)
    for i in range(START_ID, len(components)):
        component = component_map == i
        reachable_pairs += component[:, :, None, None] * component[None, None, :, :]

    possible_pairs *= reachable_pairs

    return np.stack(np.nonzero(possible_pairs), axis=-1)


class GridConfigError(BaseException):
    def __init__(self, seed):
        message = f"Could not generate enough valid start and target positions for map with seed {seed}"
        super().__init__(message)


def generate_random_grid_with_min_dist(
    seed,
    map_w,
    num_agents,
    obstacle_density,
    obs_radius,
    collision_system,
    on_target,
    min_dist,
    max_episode_steps,
    num_tries=5,
):
    rng = np.random.default_rng(seed)

    obstacles = rng.binomial(1, obstacle_density, (map_w, map_w))

    # Generating start and target positions
    possible_pairs = generate_start_target_pairs(
        obstacles=obstacles, map_w=map_w, min_dist=min_dist
    )

    for _ in range(num_tries):
        # Trying out a random permutation
        possible_pairs = rng.permutation(possible_pairs, axis=0)

        start_positions = []
        target_positions = []
        for start_x, start_y, target_x, target_y in possible_pairs.tolist():
            start_pos = [start_x, start_y]
            target_pos = [target_x, target_y]
            if (
                start_pos in start_positions
                or start_pos in target_positions
                or target_pos in start_positions
                or target_pos in target_positions
            ):
                continue
            else:
                start_positions.append(start_pos)
                target_positions.append(target_pos)
            if len(start_positions) == num_agents:
                break

        if len(start_positions) == num_agents:
            # No need to retry
            break

    if len(start_positions) < num_agents:
        # Could not find a suitable configuration
        raise GridConfigError(seed)

    return GridConfig(
        num_agents=num_agents,  # number of agents
        size=map_w,  # size of the grid
        density=obstacle_density,  # obstacle density
        seed=seed,
        max_episode_steps=max_episode_steps,  # horizon
        obs_radius=obs_radius,  # defines field of view
        observation_type="MAPF",
        collision_system=collision_system,
        on_target=on_target,
        map=obstacles.tolist(),
        agents_xy=start_positions,
        targets_xy=target_positions,
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
    min_dist,
    max_episode_steps=None,
):
    if map_type == "RandomGrid":
        assert map_h == map_w, (
            f"Expect height and width of random grid to be the same, "
            f"but got height {map_h} and width {map_w}"
        )

        if (min_dist is None) or (min_dist <= 2):

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
            # We need a custom generator to enforce the min dist between
            # start and target pos for each agent
            def _grid_config_generator(seed):
                return generate_random_grid_with_min_dist(
                    seed,
                    map_w=map_w,
                    num_agents=num_agents,
                    obstacle_density=obstacle_density,
                    obs_radius=obs_radius,
                    collision_system=collision_system,
                    on_target=on_target,
                    min_dist=min_dist,
                    max_episode_steps=max_episode_steps,
                )

    else:
        raise ValueError(f"Unsupported map type: {map_type}.")
    return _grid_config_generator
