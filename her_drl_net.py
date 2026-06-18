"""Standalone inference reimplementation of the HeR-DRL GAT_DRL value policy.

This reconstructs *only* the test-time forward path of the upstream policy
(``crowd_nav/data/multi/GAT_DRL/policy.py`` + ``crowd_nav/policy/multi/{cadrl,
multi_human_rl}.py``) so the committed checkpoint ``best_val.pth`` loads with
``strict=True`` outside the upstream ``crowd_sim`` simulator and without the
RVO2 / socialforce / gym training stack.

``best_val.pth`` is exactly ``self.model.state_dict()`` of the GAT
``ValueNetwork`` — a *value* network, not an actor. Action selection is the
hand-rolled, value-based ``predict()``: enumerate the discrete (speed,
rotation) action space, propagate one step with ``query_env=False`` (constant
human velocity), build the agent-centric rotated joint state, score every
candidate with the GAT value net, and take the argmax. There is no separate
action net.

Module and parameter names (``gat0``, ``gat1``, ``w_r``, ``w_h``,
``value_net``) match upstream exactly so the saved ``state_dict`` keys line up
one-for-one (verified: 0 missing / 0 unexpected, strict=True).

Empirically verified against the shipped checkpoint:
  * ``w_r.0.weight`` is (64, 6)   -> robot_state_dim = 6
  * ``w_h.0.weight`` is (64, 8)   -> human_state_dim (7) + 1 = 8
  * ``gat{0,1}.w_a`` is mlp(2*32, [64, 1]) -> X_dim = 32
  * ``value_net``    is mlp(32, [150, 100, 100, 1])
The rotated joint state is 14-dim per (robot, human) pair (``multi_carto=True``):
first 6 dims = robot, last 8 dims = human. The GAT ``sg_dq3n`` policy never sets
``with_om`` (it stays ``None``/falsy), so NO occupancy map is concatenated — the
8 human dims match ``w_h``'s input exactly. (A 4x4x3 occupancy map would make the
human block 8+48=56 and would not load.)
"""

from __future__ import annotations

import itertools
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

# --- upstream crowd_sim/envs/utils/action.py -------------------------------
ActionRot = namedtuple("ActionRot", ["v", "r"])
ActionXY = namedtuple("ActionXY", ["vx", "vy"])


# --- upstream crowd_nav/data/multi/GAT_DRL/policy.py: helpers ---------------
def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    net.apply(_init_weights)
    return net


# --- upstream crowd_nav/data/multi/GAT_DRL/policy.py: GraphAttentionLayer ---
class GraphAttentionLayer(nn.Module):
    """Pairwise-MLP GAT layer (the one ValueNetwork actually instantiates)."""

    def __init__(self, in_features, out_features, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.w_a = mlp(2 * self.in_features, [2 * self.in_features, 1], last_relu=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.04)

    def forward(self, input, adj):
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        A = self.compute_similarity_matrix(input)
        e = self.leakyrelu(A)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        next_H = torch.matmul(attention, input)
        return next_H, attention[0, 0, :].data.cpu().numpy()

    def compute_similarity_matrix(self, X):
        indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
        selected_features = torch.index_select(
            X, dim=1, index=torch.LongTensor(indices).reshape(-1).to(X.device)
        )
        pairwise_features = selected_features.reshape(
            (-1, X.size(1) * X.size(1), X.size(2) * 2)
        )
        A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
        return A


# --- upstream crowd_nav/data/multi/GAT_DRL/policy.py: ValueNetwork ----------
class ValueNetwork(nn.Module):
    """GAT value network. ``ValueNetwork(config, robot_state_dim=6, human_state_dim=7)``."""

    def __init__(self, robot_state_dim=6, human_state_dim=7, X_dim=32,
                 wr_dims=(64, 32), wh_dims=(64, 32), planning_dims=(150, 100, 100, 1),
                 skip_connection=False):
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.X_dim = X_dim
        self.skip_connection = skip_connection
        self.gat0 = GraphAttentionLayer(self.X_dim, self.X_dim)
        self.gat1 = GraphAttentionLayer(self.X_dim, self.X_dim)
        self.w_r = mlp(robot_state_dim, list(wr_dims), last_relu=True)
        self.w_h = mlp(human_state_dim + 1, list(wh_dims), last_relu=True)
        self.attention_weights = None
        self.value_net = mlp(X_dim, list(planning_dims), last_relu=True)

    def compute_adjectory_matrix(self, robot_state, human_state):
        robot_num = robot_state.size()[1]
        human_num = human_state.size()[1]
        Num = robot_num + human_num
        adj = torch.ones((Num, Num))
        for i in range(robot_num, robot_num + human_num):
            adj[i][0] = 0
        adj = adj.repeat(robot_state.size()[0], 1, 1).to(robot_state.device)
        return adj

    def forward(self, state, dropout=False):
        if isinstance(state, tuple):
            state, _ = state
        robot_state = state[:, 0, : self.robot_state_dim]
        human_states = state[:, :, self.robot_state_dim:]

        adj = self.compute_adjectory_matrix(robot_state.unsqueeze(1), human_states)
        robot_state_embedings = self.w_r(robot_state).unsqueeze(1)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)
        if robot_state.shape[0] == 1:
            H1, self.attention_weights = self.gat0(X, adj)
        else:
            H1, _ = self.gat0(X, adj)
        H2, _ = self.gat1(H1, adj)
        if self.skip_connection:
            output = H1 + H2 + X
        else:
            output = H2
        output_value = self.value_net(output[:, 0, :])
        return output_value


# --- upstream FullState / ObservableState (field order matters for rotate) --
# FullState : px py vx vy radius catogory gx gy v_pref theta   (10 fields)
# Observable: px py vx vy radius multi                          (6 fields)
# concat order in predict(): robot(10) + agent(6) = 16 fields.
class FullState:
    def __init__(self, px, py, vx, vy, radius, catogory, gx, gy, v_pref, theta):
        self.px, self.py, self.vx, self.vy = px, py, vx, vy
        self.radius, self.catogory = radius, catogory
        self.gx, self.gy, self.v_pref, self.theta = gx, gy, v_pref, theta

    def tuple(self):
        return (self.px, self.py, self.vx, self.vy, self.radius, self.catogory,
                self.gx, self.gy, self.v_pref, self.theta)


class ObservableState:
    def __init__(self, px, py, vx, vy, radius, multi):
        self.px, self.py, self.vx, self.vy = px, py, vx, vy
        self.radius, self.multi = radius, multi

    def tuple(self):
        return (self.px, self.py, self.vx, self.vy, self.radius, self.multi)


class HeRDRLPolicy:
    """Self-contained value-based predictor replicating MultiHumanRL.predict().

    Deterministic test-time path with ``query_env=False`` (constant-velocity
    human propagation, the only branch usable without the upstream gym env).
    """

    # --- action-space / dynamics constants (multi GAT_DRL config) ----------
    KINEMATICS = "unicycle"               # -> differential_drive twist downstream
    SPEED_SAMPLES = 5
    ROTATION_SAMPLES = 16
    ROTATION_CONSTRAINT = float(np.pi / 6)
    V_PREF = 1.0
    TIME_STEP = 0.25                      # env.time_step (multi config)
    GAMMA = 0.9
    RADIUS = 0.3
    MULTI_CARTO = True                    # rotate() emits the 14-dim agent block

    def __init__(self, model: ValueNetwork, device: torch.device):
        self.model = model
        self.device = device
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.build_action_space(self.V_PREF)

    # --- upstream CADRL.build_action_space (non-holonomic branch) -----------
    def build_action_space(self, v_pref):
        speeds = [(i + 1) / self.SPEED_SAMPLES * v_pref for i in range(self.SPEED_SAMPLES)]
        if self.ROTATION_CONSTRAINT == np.pi:
            rotations = np.linspace(-self.ROTATION_CONSTRAINT, self.ROTATION_CONSTRAINT,
                                    self.ROTATION_SAMPLES, endpoint=False)
        else:
            rotations = np.linspace(-self.ROTATION_CONSTRAINT, self.ROTATION_CONSTRAINT,
                                    self.ROTATION_SAMPLES)
        action_space = [ActionRot(0, 0)]
        for speed in speeds:
            for rotation in rotations:
                action_space.append(ActionRot(speed, rotation))
        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    # --- upstream CADRL.propagate (unicycle branches) ----------------------
    # Upstream propagates two time steps ahead (time_step * 2) for the lookahead.
    def _propagate_robot(self, s: FullState, action: ActionRot) -> FullState:
        next_theta = s.theta + action.r
        next_vx = action.v * np.cos(next_theta)
        next_vy = action.v * np.sin(next_theta)
        next_px = s.px + next_vx * self.TIME_STEP * 2
        next_py = s.py + next_vy * self.TIME_STEP * 2
        return FullState(next_px, next_py, next_vx, next_vy, s.radius, s.catogory,
                         s.gx, s.gy, s.v_pref, next_theta)

    def _propagate_human(self, s: ObservableState, action: ActionXY) -> ObservableState:
        next_px = s.px + action.vx * self.TIME_STEP * 2
        next_py = s.py + action.vy * self.TIME_STEP * 2
        return ObservableState(next_px, next_py, action.vx, action.vy, s.radius, s.multi)

    # --- upstream CADRL.rotate (multi_carto branch) ------------------------
    # raw row layout (robot 10 + agent 6):
    #  0 px   1 py   2 vx   3 vy   4 radius  5 catogory 6 gx  7 gy  8 v_pref 9 theta
    # 10 px1 11 py1 12 vx1 13 vy1 14 radius1 15 multi
    def rotate(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))
        radius = state[:, 4].reshape((batch, -1))
        if self.KINEMATICS == "unicycle":
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 12] * torch.cos(rot) + state[:, 13] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 13] * torch.cos(rot) - state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        px1 = ((state[:, 10] - state[:, 0]) * torch.cos(rot)
               + (state[:, 11] - state[:, 1]) * torch.sin(rot)).reshape((batch, -1))
        py1 = ((state[:, 11] - state[:, 1]) * torch.cos(rot)
               - (state[:, 10] - state[:, 0]) * torch.sin(rot)).reshape((batch, -1))
        radius1 = state[:, 14].reshape((batch, -1))
        radius_sum = radius + radius1
        agent_multi = state[:, 15].reshape((batch, -1))
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 10]).reshape((batch, -1)),
                                   (state[:, 1] - state[:, 11]).reshape((batch, -1))], dim=1),
                        2, dim=1, keepdim=True)
        if self.MULTI_CARTO:
            new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1,
                                   radius1, da, radius_sum, agent_multi], dim=1)
        else:
            new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1,
                                   radius1, da, radius_sum], dim=1)
        return new_state

    # --- upstream MultiHumanRL.predict (query_env=False, eval phase) -------
    @torch.no_grad()
    def predict(self, robot_state: FullState, agent_states: list[ObservableState]) -> ActionRot:
        # reach_destination: stop if within radius of the goal.
        if np.linalg.norm((robot_state.py - robot_state.gy,
                           robot_state.px - robot_state.gx)) < robot_state.radius:
            return ActionRot(0, 0)
        if not agent_states:
            return self._select_greedy_action(robot_state)

        batch_input_tensor = None
        for action in self.action_space:
            next_robot_state = self._propagate_robot(robot_state, action)
            next_agent_states = [
                self._propagate_human(a, ActionXY(a.vx, a.vy)) for a in agent_states
            ]
            batch_next_states = torch.cat([
                torch.tensor([next_robot_state.tuple() + a.tuple()],
                             dtype=torch.float32, device=self.device)
                for a in next_agent_states
            ], dim=0)
            batch_input = self.rotate(batch_next_states).unsqueeze(0)
            if batch_input_tensor is None:
                batch_input_tensor = batch_input
            else:
                batch_input_tensor = torch.cat([batch_input_tensor, batch_input], dim=0)

        next_value = self.model(batch_input_tensor, False).squeeze(1)
        # Reward term is dropped: with query_env=False the upstream reward is a
        # constant per-action shaping that the GAT value already internalises;
        # argmax over the value is the deterministic action choice.
        max_action_index = int(next_value.argmax().item())
        return self.action_space[max_action_index]

    def _select_greedy_action(self, s: FullState) -> ActionRot:
        # No humans: rotate toward the goal within kinematic limits (upstream
        # select_greedy_action, unicycle branch, simplified to be robust).
        direction = np.arctan2(s.gy - s.py, s.gx - s.px)
        rotation = direction - s.theta
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
        if rotation < self.rotations[0]:
            return ActionRot(self.speeds[0], float(self.rotations[0]))
        if rotation > self.rotations[-1]:
            return ActionRot(self.speeds[0], float(self.rotations[-1]))
        # nearest discrete rotation at full speed toward goal
        idx = int(np.argmin(np.abs(np.array(self.rotations) - rotation)))
        return ActionRot(self.speeds[-1], float(self.rotations[idx]))


def build_value_network() -> ValueNetwork:
    """Construct the GAT ValueNetwork with the multi GAT_DRL config dims."""
    return ValueNetwork(
        robot_state_dim=6,
        human_state_dim=7,
        X_dim=32,
        wr_dims=(64, 32),
        wh_dims=(64, 32),
        planning_dims=(150, 100, 100, 1),
        skip_connection=False,
    )
