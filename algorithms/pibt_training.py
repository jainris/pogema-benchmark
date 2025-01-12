import argparse
import pickle
import pathlib
import numpy as np

from pogema import pogema_v0, GridConfig

from grid_config_generator import add_grid_config_args, grid_config_generator_factory

from pibt.pypibt.pibt import PIBT
from pogema_toolbox.algorithm_config import AlgoBase
from collision_shielding import get_neighbors

from run_expert import add_expert_dataset_args, get_expert_dataset_file_name


class PIBTSaver(PIBT):
    def __init__(self, grid, starts, goals, moves, seed=0):
        super().__init__(grid, starts, goals, seed)
        self.moves = moves
        self.saved_data = []

    def funcPIBT(self, Q_from, Q_to, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C, move_idx, mask = get_neighbors(self.grid, Q_from[i], self.moves)
        for j in range(len(move_idx)):
            self.saved_data[-1][i][move_idx[j]] = self.dist_tables[i].get(C[j])

        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(self, Q_from, priorities: list[float]):
        # setup
        N = len(Q_from)
        Q_to = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # Invalid moves get default value: -1 (as 0 is reserved for the goal)
        self.saved_data.append(-np.ones((N, len(self.moves)), dtype=np.int))

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to


class PIBTInferenceConfig(AlgoBase):
    pass


class PIBTInference:
    def __init__(self, config=None, env=None):
        self.env = env
        self.config = config
        self.output_data = None
        self.step = 1
        self.env = env
        self.pibt_expert = None
        if env is not None:
            self.MOVES = np.array(self.env.grid_config.MOVES)

    def reset_states(self, env=None):
        self.step = 1
        self.timed_out = False
        self.pibt_expert = None
        if env is not None:
            self.env = env
            self.MOVES = np.array(self.env.grid_config.MOVES)

    def run_pibt(self):
        obstacles = self.env.grid.get_obstacles(ignore_borders=True)
        starts = self.env.grid.get_agents_xy(ignore_borders=True)
        goals = self.env.grid.get_targets_xy(ignore_borders=True)

        starts = [tuple(s) for s in starts]
        goals = [tuple(g) for g in goals]

        self.pibt_expert = PIBTSaver(
            obstacles == 0,
            starts,
            goals,
            moves=self.env.grid_config.MOVES,
            seed=self.env.grid_config.seed,
        )
        return self.pibt_expert.run(self.env.grid.config.max_episode_steps + 1)

    def _get_next_move(self, step):
        if step >= len(self.output_data):
            return [0] * self.env.grid_config.num_agents
        diff = self.output_data[step] - self.output_data[step - 1]
        actions = np.argmax(
            np.all(self.MOVES[None, :, :] == diff[:, None, :], axis=-1), axis=-1
        )
        return actions.tolist()

    def act(
        self, observations=None, rewards=None, dones=None, info=None, skip_agents=None
    ):
        if self.output_data is None:
            output_data = self.run_pibt()
            self.output_data = np.array(output_data)
        actions = self._get_next_move(self.step)
        self.step += 1
        return actions


def get_expert_algorithm_and_config(args):
    if args.expert_algorithm == "PIBT":
        inference_config = PIBTInferenceConfig()
        expert_algorithm = PIBTInference
    else:
        raise ValueError(f"Unsupported expert algorithm {args.expert_algorithm}.")
    return expert_algorithm, inference_config


def get_relevance_from_expert(expert_algorithm):
    if isinstance(expert_algorithm, PIBTInference):
        inverse_relevance = np.stack(expert_algorithm.pibt_expert.saved_data, axis=0)
        max_dists = np.max(inverse_relevance, keepdims=True, axis=-1)

        # Adding 1 to max_dists, so that relevance for max dist cell is 1,
        # as 0 is reserved for invalid moves
        max_dists = max_dists + 1
        invalid_moves = inverse_relevance < 0

        relevance = max_dists - inverse_relevance
        relevance[invalid_moves] = 0

        return relevance
    else:
        raise ValueError(
            f"Unsupported expert_algorithm type: {type(expert_algorithm)}."
        )


def run_expert_algorithm(
    expert,
    env=None,
    observations=None,
    grid_config=None,
    save_termination_state=True,
    additional_data_func=None,
):
    if env is None:
        env = pogema_v0(grid_config=grid_config)
        observations, infos = env.reset()

    all_actions = []
    all_observations = []
    all_terminated = []
    additional_data = []

    expert.reset_states(env)

    while True:
        actions = expert.act(observations)

        all_observations.append(observations)
        all_actions.append(actions)

        if additional_data_func is not None:
            additional_data.append(
                additional_data_func(
                    env=env, observations=observations, actions=actions
                )
            )

        observations, rewards, terminated, truncated, infos = env.step(actions)

        if save_termination_state:
            all_terminated.append(terminated)

        if all(terminated) or all(truncated):
            break

    relevance = get_relevance_from_expert(expert)

    if additional_data_func is not None:
        return all_actions, all_observations, all_terminated, relevance, additional_data
    return all_actions, all_observations, all_terminated, relevance


def main():
    parser = argparse.ArgumentParser(description="Run PIBT Expert.")
    parser = add_expert_dataset_args(parser)

    args = parser.parse_args()
    print(args)

    assert args.pibt_expert_relevance_training

    num_agents = int(args.robot_density * args.map_h * args.map_w)

    rng = np.random.default_rng(args.dataset_seed)
    seeds = rng.integers(10**10, size=args.num_samples)

    _grid_config_generator = grid_config_generator_factory(
        map_type=args.map_type,
        map_w=args.map_w,
        map_h=args.map_h,
        num_agents=num_agents,
        obstacle_density=args.obstacle_density,
        obs_radius=args.obs_radius,
        collision_system=args.collision_system,
        on_target=args.on_target,
        min_dist=args.min_dist,
        max_episode_steps=args.max_episode_steps,
    )

    expert_algorithm, inference_config = get_expert_algorithm_and_config(args)

    dataset = []
    all_relevance = []
    seed_mask = []
    num_success = 0
    for i, seed in enumerate(seeds):
        grid_config = _grid_config_generator(seed)
        print(f"Running expert on map {i + 1}/{args.num_samples}", end=" ")
        expert = expert_algorithm(inference_config)

        all_actions, all_observations, all_terminated, relevance = run_expert_algorithm(
            expert,
            grid_config=grid_config,
            save_termination_state=args.save_termination_state,
        )

        if all(all_terminated[-1]):
            seed_mask.append(True)
            num_success += 1
            if args.save_termination_state:
                dataset.append((all_observations, all_actions, all_terminated))
            else:
                dataset.append((all_observations, all_actions))
            all_relevance.append(relevance)
        else:
            seed_mask.append(False)

        print(f"-- Success Rate: {num_success / (i + 1)}")

    print(f"{len(dataset)}/{len(seeds)} samples were successfully added to the dataset")

    file_name = get_expert_dataset_file_name(args)
    path = pathlib.Path(f"{args.dataset_dir}", "raw_expert_predictions", f"{file_name}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((dataset, seed_mask), f)

    path = pathlib.Path(f"{args.dataset_dir}", "pibt_relevance", f"{file_name}")
    all_relevance = np.concatenate(all_relevance, axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(all_relevance, f)


if __name__ == "__main__":
    main()
