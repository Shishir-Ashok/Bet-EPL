"""
backend/model/dqn_agent.py
---------------------------
Deep Q-Network (DQN) betting agent.

Architecture overview:
  XGBoost (outcome probabilities)
       ↓
  DQN state = [16 features incl. XGBoost probs]
       ↓
  Q-Network → Q(s, a) for each action
       ↓
  action = argmax Q(s, a)  [or random with prob ε during training]

Action space (4 actions):
  0 = BET_HOME   — bet on home win
  1 = BET_DRAW   — bet on draw
  2 = BET_AWAY   — bet on away win
  3 = PASS       — skip this match

Reward design (see train.py compute_reward for full details):
  Historical training uses a flat 1 EUR stake so the agent receives a
  meaningful signal regardless of odds source availability.
    WIN  →  +(odds - 1.0)   e.g. +1.0 to +5.0
    LOSS →  -1.0
    PASS →   0.0  (neutral — neither rewarded nor penalised)
  Rewards are clipped to [-2, +2] before storage to prevent rare
  high-odds outliers from dominating the loss.

Epsilon schedule:
  EPS_DECAY = 0.905 per epoch so that after 30 epochs ε reaches 0.05
  (full exploit). With the previous value of 0.995, ε was still 0.86
  after 30 epochs — effectively still random throughout training.
"""

import os
import sys
import json
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.db import supabase
from backend.model.features import DQN_STATE_DIM

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
os.makedirs(MODEL_DIR, exist_ok=True)

ACTION_DIM   = 4
ACTION_NAMES = {0: "BET_HOME", 1: "BET_DRAW", 2: "BET_AWAY", 3: "PASS"}

# Reduced from 1e-3 → 3e-4.
# 1e-3 overshoots the tiny reward signal and lands at a flat minimum
# in one epoch. 3e-4 gives stable convergence on small datasets.
LEARNING_RATE = 3e-4

GAMMA = 0.95

# Sync target net every 20 steps (was 50).
# With ~1,500 matches per epoch and batch_size=64, one epoch = ~23 updates.
# Syncing every 20 keeps the target fresh without instability.
TARGET_UPDATE_FREQ = 20

GRAD_CLIP = 1.0

# Epsilon schedule: decays so ε reaches EPS_END after ~30 epochs.
# Formula: EPS_END = EPS_START * EPS_DECAY^30  →  EPS_DECAY = (0.05)^(1/30) ≈ 0.905
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 0.905


# ─── Network ──────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Q-value function approximator.

    Dropout removed: with 1,482 training samples and rewards in [-2, +2],
    0.2 dropout was zeroing ~300 neurons per forward pass and making
    gradients vanish. BatchNorm instead provides regularisation.
    """

    def __init__(self, state_dim: int = DQN_STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Agent ────────────────────────────────────────────────────────────────────

class DQNAgent:

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"  DQN device: {self.device}")

        self.online_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.epsilon     = EPS_START
        self.train_steps = 0
        self.losses      = []

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(ACTION_DIM)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.online_net(state_t).argmax(dim=1).item())

    def get_q_values(self, state: np.ndarray) -> dict:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t)[0].cpu().numpy()
        return {ACTION_NAMES[i]: round(float(q_vals[i]), 4) for i in range(ACTION_DIM)}

    def train_step(self, batch: dict) -> float:
        """
        Double DQN update:
          - Online net selects the next action (reduces overestimation)
          - Target net evaluates it (provides stable targets)
        """
        states      = torch.FloatTensor(batch["states"]).to(self.device)
        actions     = torch.LongTensor(batch["actions"]).to(self.device)
        rewards     = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones       = torch.FloatTensor(batch["dones"]).to(self.device)

        # Set online net to eval for target computation (no LayerNorm training stats)
        self.online_net.eval()
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        self.online_net.train()
        current_q_train = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current_q_train, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        self.train_steps += 1
        loss_val = float(loss.item())
        self.losses.append(loss_val)

        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss_val

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def get_confidence(self, state: np.ndarray) -> float:
        """Returns how much better the best action is vs PASS (Q-advantage)."""
        q_vals = self.get_q_values(state)
        best_action = max(q_vals, key=q_vals.get)
        if best_action == "PASS":
            return 0.0
        return round(max(0.0, q_vals[best_action] - q_vals["PASS"]), 4)

    def save(self, version_tag: str) -> str:
        path = os.path.join(MODEL_DIR, f"{version_tag}.pt")
        torch.save({
            "online_net":  self.online_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "train_steps": self.train_steps,
        }, path)
        print(f"  DQN saved: {path}")
        return path

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon     = checkpoint.get("epsilon", EPS_END)
        self.train_steps = checkpoint.get("train_steps", 0)
        print(f"  DQN loaded: {path} (ε={self.epsilon:.3f}, steps={self.train_steps})")

    @classmethod
    def load_active(cls) -> "DQNAgent":
        from backend.model.model_store import ensure_local

        result = (
            supabase.table("model_versions")
            .select("version_tag, storage_path")
            .eq("model_type", "dqn")
            .eq("is_active", True)
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise FileNotFoundError(
                "No active DQN model. Run: python -m backend.model.train --mode dqn"
            )

        storage_path = result.data[0]["storage_path"]
        filename     = storage_path.split("/")[-1]
        local_path   = os.path.join(MODEL_DIR, filename)

        # Downloads from Supabase Storage if local file was wiped by Render
        local_path = ensure_local(storage_path, local_path)

        agent = cls()
        agent.load(local_path)
        return agent


# ─── DB helpers ───────────────────────────────────────────────────────────────

def _sanitise(obj):
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _register_dqn(version_tag, save_path, metrics, avg_reward):
    from backend.model.model_store import upload

    storage_path = upload(save_path)

    supabase.table("model_versions").update(
        {"is_active": False}
    ).eq("model_type", "dqn").eq("is_active", True).execute()

    supabase.table("model_versions").upsert({
        "version_tag":  version_tag,
        "model_type":   "dqn",
        "avg_reward":   round(float(avg_reward), 4),
        "is_active":    True,
        "storage_path": storage_path,
        "notes":        json.dumps(_sanitise(metrics)),
    }, on_conflict="version_tag").execute()

    print(f"  Registered DQN version: {version_tag} (is_active=True)")