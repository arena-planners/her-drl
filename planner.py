"""HeR-DRL (GAT_DRL, unicycle) wrapper for the arena_planners bridge.

Reconstructs, outside the upstream ``crowd_sim`` simulator, the per-(robot,
human) joint state and runs the value-based GAT policy
(``crowd_nav/data/multi/GAT_DRL``) for a SINGLE robot. The committed checkpoint
``model/best_val.pth`` is exactly ``self.model.state_dict()`` of the GAT
``ValueNetwork`` (a value model, not an actor); action selection is the
hand-rolled, deterministic ``predict()`` that enumerates the discrete action
space, propagates one step with ``query_env=False``, scores each candidate with
the value net, and takes the argmax. See ``her_drl_net.py``.

The selected ``ActionRot(v, r)`` is a unicycle action: ``v`` is the commanded
forward speed and ``r`` is the change of heading applied over one ``time_step``.
It maps to a differential-drive ``[v, omega]`` twist with ``omega = r /
time_step`` — no extra diff-drive projection is performed here.

No observation normalization is applied: upstream evaluation consumes raw
states (no running mean/var / ob_rms ships with the checkpoint).
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from arena_planners.sdk import load_manifest, main_loop

from her_drl_net import (
    FullState,
    HeRDRLPolicy,
    ObservableState,
    build_value_network,
)

_HERE = pathlib.Path(__file__).parent
_CHECKPOINT = _HERE / "model" / "best_val.pth"

# Constants from the multi GAT_DRL config (config_multi.py / data/multi/GAT_DRL).
_RADIUS: float = 0.3                # robot.radius
_V_PREF: float = 1.0                # robot.v_pref
_TIME_STEP: float = 0.25            # env.time_step


class _Runner:
    def __init__(self) -> None:
        # Device-agnostic: prefer CUDA only if actually available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = build_value_network()
        state = torch.load(str(_CHECKPOINT), map_location="cpu", weights_only=False)
        net.load_state_dict(state, strict=True)  # 0 missing / 0 unexpected (verified)
        net.eval()
        net.to(self.device)
        self.policy = HeRDRLPolicy(net, self.device)

    def act(self, robot: FullState, humans: list[ObservableState]):
        return self.policy.predict(robot, humans)


_runner: _Runner | None = None


def _get_runner() -> _Runner:
    global _runner
    if _runner is None:
        _runner = _Runner()
    return _runner


def step(features: dict) -> list[float]:
    """Map the bridge feature dict to a differential-drive [v, omega] twist."""
    runner = _get_runner()

    robot_pose = features.get("robot_pose")
    goal_pose = features.get("goal_pose")
    if robot_pose is None or goal_pose is None:
        return [0.0, 0.0]

    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    robot_state = features.get("robot_state")
    if robot_state is not None and len(robot_state) >= 4:
        vx, vy = float(robot_state[2]), float(robot_state[3])
    else:
        vx, vy = 0.0, 0.0
    gx, gy = float(goal_pose[0]), float(goal_pose[1])

    # robot FullState: catogory=0 marks the robot (vs 1 for humans).
    robot = FullState(px, py, vx, vy, _RADIUS, 0.0, gx, gy, _V_PREF, theta)

    humans: list[ObservableState] = []
    peds = features.get("pedestrians")
    if peds is not None:
        for ped in peds:
            # bridge pedestrian row: [id, px, py, vx, vy, ...]
            hx, hy = float(ped[1]), float(ped[2])
            hvx = float(ped[3]) if len(ped) > 3 else 0.0
            hvy = float(ped[4]) if len(ped) > 4 else 0.0
            humans.append(ObservableState(hx, hy, hvx, hvy, _RADIUS, 1.0))

    action = runner.act(robot, humans)  # ActionRot(v, r)

    v = float(action.v)
    # r is a per-time_step heading change -> angular velocity.
    omega = float(action.r) / _TIME_STEP
    return [v, omega]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    # Stateless value-based policy: nothing to reset (action space is fixed).
    _get_runner()


if __name__ == "__main__":
    manifest = load_manifest(_HERE / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
