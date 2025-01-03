import numpy as np
import pathlib

from typing import Literal
from pogema_toolbox.algorithm_config import AlgoBase

from pogema import GridConfig

import os
import subprocess

import ruamel.yaml

yaml = ruamel.yaml.YAML()

lib_path = os.path.join(os.path.dirname(__file__), "build", "ecbs")
lib_path = pathlib.Path(lib_path)
lib_path.parent.mkdir(parents=True, exist_ok=True)

if not os.path.exists(lib_path):
    calling_script_dir = os.path.dirname(lib_path)
    cmake_cmd = ["cmake", "../libMultiRobotPlanning"]
    subprocess.run(cmake_cmd, check=True, cwd=calling_script_dir)
    make_cmd = ["make", "ecbs"]
    subprocess.run(make_cmd, check=True, cwd=calling_script_dir)

DEFAULT_TMP = os.path.join(os.path.dirname(__file__), "tmp")


class ECBSInferenceConfig(AlgoBase):
    name: Literal["ECBS"] = "ECBS"
    suboptimality: float = 1.0
    disappear_at_goal: bool = False
    tmp_dir: str = DEFAULT_TMP
    timeout: float = 60.0


class ECBSLib:
    def __init__(self, config: ECBSInferenceConfig):
        self.config = config
        tmp_dir = config.tmp_dir
        self.input_file = os.path.abspath(os.path.join(tmp_dir, "input.yaml"))
        self.output_file = os.path.abspath(os.path.join(tmp_dir, "output.yaml"))
        tmp_dir = pathlib.Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    def prepare_input(self, env):
        start_locs = env.grid.get_agents_xy(ignore_borders=True)
        target_locs = env.grid.get_targets_xy(ignore_borders=True)
        obstacle_locs = np.stack(
            np.nonzero(env.grid.get_obstacles(ignore_borders=True))
        ).T

        input_data = {"agents": [], "map": {"dimensions": [], "obstacles": []}}
        for agent_id, (start, goal) in enumerate(zip(start_locs, target_locs)):
            s, g = ruamel.yaml.comments.CommentedSeq(
                start
            ), ruamel.yaml.comments.CommentedSeq(goal)
            s.fa.set_flow_style()
            g.fa.set_flow_style()
            input_data["agents"].append(
                {"start": s, "goal": g, "name": f"agent{agent_id}"}
            )
        input_data["map"]["dimensions"] = ruamel.yaml.comments.CommentedSeq(
            [env.grid_config.size, env.grid_config.size]
        )
        input_data["map"]["dimensions"].fa.set_flow_style()
        for obstacle in obstacle_locs:
            o = ruamel.yaml.comments.CommentedSeq(obstacle.tolist())
            o.fa.set_flow_style()
            input_data["map"]["obstacles"].append(o)
        with open(self.input_file, "w") as f:
            yaml.dump(input_data, f)
        return input_data

    def parse_output(self):
        with open(self.output_file, "r") as f:
            output_data = yaml.load(f)
        return output_data

    def run_ecbs(self, env):
        self.prepare_input(env)

        calling_script_dir = lib_path.parent
        ecbs_command = [
            "./ecbs",
            "-i",
            self.input_file,
            "-o",
            self.output_file,
            "-w",
            str(self.config.suboptimality),
        ]
        if self.config.disappear_at_goal:
            ecbs_command.append("--disappear-at-goal")

        try:
            subprocess.run(
                ecbs_command,
                check=True,
                cwd=calling_script_dir,
                timeout=self.config.timeout,
                stdout=subprocess.DEVNULL,
            )
        except TimeoutError:
            return None

        return self.parse_output()


class ECBSInference:
    def __init__(self, config: ECBSInferenceConfig, env=None):
        self.config = config
        self.ecbs_lib = ECBSLib(config)
        self.output_data = None
        self.step = 1
        self.env = env
        if env is not None:
            self.MOVES = np.array(self.env.grid_config.MOVES)
        self.timed_out = False

    def reset_states(self, env=None):
        self.step = 1
        self.timed_out = False
        if env is not None:
            self.env = env
            self.MOVES = np.array(self.env.grid_config.MOVES)

    def _get_pos_from_idx(self, agent_id, idx):
        return np.array(
            [
                self.output_data["schedule"][f"agent{agent_id}"][idx]["x"],
                self.output_data["schedule"][f"agent{agent_id}"][idx]["y"],
            ]
        )

    def _get_next_move_single_agent(self, agent_id, step):
        idx = 0
        for data in self.output_data["schedule"][f"agent{agent_id}"]:
            if data["t"] >= step:
                break
            idx += 1
        if idx == len(self.output_data["schedule"][f"agent{agent_id}"]):
            return 0
        if self.output_data["schedule"][f"agent{agent_id}"][idx]["t"] == step:
            new_pos = self._get_pos_from_idx(agent_id, idx)
            old_pos = self._get_pos_from_idx(agent_id, idx - 1)
            return np.nonzero(np.all(self.MOVES == (new_pos - old_pos), axis=-1))[0][0]
        else:
            return 0

    def _get_next_move(self, step):
        return [
            self._get_next_move_single_agent(agent_id, step)
            for agent_id in range(self.env.grid_config.num_agents)
        ]

    def act(
        self, observations=None, rewards=None, dones=None, info=None, skip_agents=None
    ):
        if self.output_data is None:
            if not self.timed_out:
                self.output_data = self.ecbs_lib.run_ecbs(self.env)
                if self.output_data is None:
                    self.timed_out = True
                    return [0] * self.env.grid_config.num_agents
            else:
                # If timed out, then just waiting (maybe change to something else?)
                return [0] * self.env.grid_config.num_agents
        actions = self._get_next_move(self.step)
        self.step += 1
        return actions
