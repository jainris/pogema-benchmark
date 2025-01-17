import torch
import numpy as np
from pibt.pypibt.pibt import PIBT
from pibt.pypibt.mapf_utils import is_valid_coord


class BaseCollisionShielding:
    def __init__(self, model, env, sampling_method="deterministic"):
        self.model = model
        self.env = env
        self.sampling_method = sampling_method

    def in_simulation(self, value):
        self.model.in_simulation(value)

    def get_actions(self, gdata):
        raise NotImplementedError


class NaiveCollisionShielding(BaseCollisionShielding):
    def __init__(self, model, env, sampling_method="deterministic"):
        super().__init__(model, env, sampling_method)
        if self.sampling_method == "probabilistic":
            self.rng = np.random.default_rng(seed=env.grid_config.seed)

    def shield(self, actions):
        if self.sampling_method == "deterministic":
            actions = torch.argmax(actions, dim=-1).detach().cpu()
        elif self.sampling_method == "probabilistic":
            probs = torch.nn.functional.softmax(actions, dim=-1)
            probs = probs.detach().cpu().numpy()

            # Despite using softmax, sum might not be 1 due to fp errors
            probs = probs / np.sum(probs, keepdims=True, axis=-1)

            actions = np.zeros(probs.shape[0], dtype=np.int)
            ids = np.arange(probs.shape[1])
            for i in range(probs.shape[0]):
                actions[i] = self.rng.choice(
                    ids, size=1, replace=False, p=probs[i], shuffle=False
                )
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}.")
        return actions

    def get_actions(self, gdata):
        # Naive collision shielding leaves the shielding to the env
        # So just returning the actions given by the model
        actions = self.model(gdata.x, gdata)
        return self.shield(actions)


def get_neighbors(grid, coord, moves):
    # coord: y, x
    neigh = []
    move_idx = []
    mask = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh, move_idx

    y, x = coord

    for i, (dy, dx) in enumerate(moves):
        if is_valid_coord(grid, (y + dy, x + dx)):
            neigh.append((y + dy, x + dx))
            move_idx.append(i)
            mask.append(True)
        else:
            mask.append(False)

    return neigh, move_idx, mask


class PIBTInstance(PIBT):
    def __init__(self, grid, starts, goals, moves, sampling_method, seed=0):
        super().__init__(grid, starts, goals, seed)

        # Calculating initial priorities
        self.priorities: list[float] = []
        for i in range(self.N):
            self.priorities.append(
                self.dist_tables[i].get(self.starts[i]) / self.grid.size
            )

        self.state = self.starts
        self.reached_goals = False
        self.moves = moves
        self.sampling_method = sampling_method

    def _update_priorities(self):
        flg_fin = True
        for i in range(self.N):
            if self.state[i] != self.goals[i]:
                flg_fin = False
                self.priorities[i] += 1
            else:
                self.priorities[i] -= np.floor(self.priorities[i])
        self.reached_goals = flg_fin

    def funcPIBT(self, Q_from, Q_to, i: int, transition_probabilities) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C, move_idx, mask = get_neighbors(self.grid, Q_from[i], self.moves)

        if self.sampling_method == "deterministic":
            ids = np.arange(len(C))
            self.rng.shuffle(ids)  # tie-breaking, randomize
            ids = sorted(
                ids,
                key=lambda u: transition_probabilities[i][move_idx[u]],
                reverse=True,
            )
        elif self.sampling_method == "probabilistic":
            cur_trans_probs = transition_probabilities[i][mask]
            cur_trans_probs = cur_trans_probs / np.sum(cur_trans_probs)

            try:
                ids = np.arange(len(C))
                ids = self.rng.choice(
                    ids, size=len(C), replace=False, p=cur_trans_probs, shuffle=False
                )
            except:
                # Potential error due to zeroing of some probs
                EPSILON = 1e-6
                cur_trans_probs = cur_trans_probs + EPSILON
                cur_trans_probs = cur_trans_probs / np.sum(cur_trans_probs)

                ids = np.arange(len(C))
                ids = self.rng.choice(
                    ids, size=len(C), replace=False, p=cur_trans_probs, shuffle=False
                )
        else:
            raise ValueError(f"Unsupported sampling method: {self.sampling_method}.")

        # vertex assignment
        for id in ids:
            v = C[id]
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.actions[i] = move_idx[id]
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j, transition_probabilities))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.actions[i] = 0
        self.occupied_nxt[Q_from[i]] = i
        return False

    def _step(self, Q_from, priorities, transition_probabilities):
        # setup
        N = len(Q_from)
        Q_to = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i, transition_probabilities)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        return Q_to

    def step(self, transition_probabilities):
        self.actions = np.zeros(self.N, dtype=np.int)
        if self.reached_goals:
            return self.actions
        self.state = self._step(self.state, self.priorities, transition_probabilities)
        self._update_priorities()
        return self.actions

    def run(self, max_timestep=1000):
        raise AssertionError("This method should not be run.")


class PIBTInstanceDist(PIBTInstance):
    def __init__(self, grid, starts, goals, moves, sampling_method, seed=0):
        super().__init__(grid, starts, goals, moves, sampling_method, seed)
        self._update_priorities()

    def _update_priorities(self):
        # Setting priorities based on distance to goal
        for i in range(self.N):
            sx, sy = self.state[i]
            gx, gy = self.goals[i]
            self.priorities[i] = abs(gx - sx) + abs(gy - sy)


class PIBTCollisionShielding(BaseCollisionShielding):
    def __init__(
        self, model, env, sampling_method="deterministic", dist_priorities=False
    ):
        super().__init__(model, env, sampling_method)

        obstacles = env.grid.get_obstacles(ignore_borders=True)
        starts = env.grid.get_agents_xy(ignore_borders=True)
        goals = env.grid.get_targets_xy(ignore_borders=True)

        starts = [tuple(s) for s in starts]
        goals = [tuple(g) for g in goals]

        if dist_priorities:
            self.pibt_instance = PIBTInstanceDist(
                grid=obstacles == 0,
                starts=starts,
                goals=goals,
                moves=env.grid_config.MOVES,
                seed=env.grid_config.seed,
                sampling_method=sampling_method,
            )
        else:
            self.pibt_instance = PIBTInstance(
                grid=obstacles == 0,
                starts=starts,
                goals=goals,
                moves=env.grid_config.MOVES,
                seed=env.grid_config.seed,
                sampling_method=sampling_method,
            )

    def get_actions(self, gdata):
        actions = self.model(gdata.x, gdata)
        if self.sampling_method == "probabilistic":
            actions = torch.nn.functional.softmax(actions, dim=-1)
            actions = actions.detach().cpu().numpy()
        actions = self.pibt_instance.step(actions)
        return actions


def get_collision_shielded_model(model, env, args):
    collision_shielding = "naive"
    if "collision_shielding" in vars(args):
        collision_shielding = args.collision_shielding
    if collision_shielding == "naive":
        return NaiveCollisionShielding(
            model=model, env=env, sampling_method=args.action_sampling
        )
    elif collision_shielding == "pibt":
        return PIBTCollisionShielding(
            model=model,
            env=env,
            sampling_method=args.action_sampling,
            dist_priorities=False,
        )
    elif collision_shielding == "pibt-dist":
        return PIBTCollisionShielding(
            model=model,
            env=env,
            sampling_method=args.action_sampling,
            dist_priorities=True,
        )
    else:
        raise ValueError(
            f"Unsupported collision shielding method: {collision_shielding}."
        )
