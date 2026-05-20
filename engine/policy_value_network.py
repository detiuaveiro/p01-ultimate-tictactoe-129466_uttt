"""
Policy-Value Network for AlphaZero-lite UTTT.

A convolutional neural network that takes a UTTT board state as input
and outputs a policy vector (move probabilities over 81 cells) and a
value scalar (position evaluation in [-1, 1]).

Architecture (inspired by AlphaZero but scaled for UTTT):
  - Input: 3 x 9 x 9 (channels: P1 stones, P2 stones, metadata)
  - Initial conv: 3 -> 160 channels, 3x3, padding=1
  - 10 residual blocks (each: Conv2d 3x3 -> BN -> ReLU -> Conv2d 3x3 -> BN -> + skip -> ReLU)
  - Policy head: Conv2d 160->8 (1x1) -> BN -> ReLU -> Flatten -> Linear 648->81
  - Value head: Conv2d 160->1 (1x1) -> BN -> ReLU -> Flatten -> Linear 81->64 -> ReLU -> Linear 64->1 -> Tanh
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.game_state import UTTTState


# ---------------------------------------------------------------------------
# ResNetBlock
# ---------------------------------------------------------------------------


class ResNetBlock(nn.Module):
    """Residual block for Policy-Value Network.

    Two convolutional layers (3x3, padding=1) with batch normalisation
    and a skip connection::

        input -> Conv2d -> BN -> ReLU -> Conv2d -> BN -> +ReLU -> output
          |                                                     ^
          +------------------- skip connection -----------------+
    """

    def __init__(self, channels: int = 160) -> None:
        """Initialise the residual block.

        Args:
            channels: Number of input/output feature channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch, channels, 9, 9).

        Returns:
            Output tensor of same shape as input.
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


# ---------------------------------------------------------------------------
# PolicyValueNetwork
# ---------------------------------------------------------------------------


class PolicyValueNetwork(nn.Module):
    """Policy-Value Network for Ultimate Tic-Tac-Toe.

    Takes a 3 x 9 x 9 board representation and outputs:
      - policy_logits: (batch, 81) raw logits for each cell.
      - value: (batch, 1) scalar in [-1, 1].
    """

    def __init__(self, channels: int = 160, num_res_blocks: int = 10) -> None:
        """Initialise the network.

        Args:
            channels: Number of feature channels in the residual tower.
            num_res_blocks: Number of residual blocks to stack.
        """
        super().__init__()
        self.channels = channels
        self.num_res_blocks = num_res_blocks

        # ---- Initial convolutional block ----
        self.conv_input = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        # ---- Residual tower ----
        self.res_blocks = nn.ModuleList(
            [ResNetBlock(channels) for _ in range(num_res_blocks)]
        )

        # ---- Policy head ----
        self.policy_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * 9 * 9, 81)

        # ---- Value head ----
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode board state, produce policy and value.

        Args:
            x: Input tensor of shape (batch, 3, 9, 9).

        Returns:
            Tuple of (policy_logits, value):
              - policy_logits: (batch, 81) raw logits.
              - value: (batch, 1) scalar in [-1, 1].
        """
        # ---- Initial convolution ----
        x = F.relu(self.bn_input(self.conv_input(x)))

        # ---- Residual tower ----
        for block in self.res_blocks:
            x = block(x)

        # ---- Policy head ----
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # flatten
        policy_logits = self.policy_fc(policy)

        # ---- Value head ----
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------


def encode_state(state: UTTTState) -> torch.Tensor:
    """Encode a UTTTState as a 3 x 9 x 9 tensor for the neural network.

    Uses vectorised operations for speed.

    Channel layout:
      - Channel 0: Current player's stones (1.0 where current player occupies).
      - Channel 1: Opponent's stones (1.0 where opponent occupies).
      - Channel 2: Metadata:
          * 1.0 in the active macro-board region (if ``active_macro`` is set).
          * Uniform fill of 1/9 in ALL unresolved boards (if ``active_macro``
            is ``None`` / free move).
          * 0.0 elsewhere.
          * Additional offset: +0.25 if current_player == 1, -0.25 if
            current_player == 2 (added to the *entire* channel).

    Args:
        state: The UTTT game state to encode.

    Returns:
        A float32 tensor of shape (3, 9, 9).
    """
    import numpy as np

    board_arr = np.array(state._board, dtype=np.int8)  # (9, 9)
    macro_arr = np.array(state._macro_board, dtype=np.int8)  # (3, 3)
    active_macro = state._active_macro
    current_player = state.current_player
    opponent = 3 - current_player

    channels = np.zeros((3, 9, 9), dtype=np.float32)

    # Channel 0: current player's stones
    channels[0] = (board_arr == current_player).astype(np.float32)
    # Channel 1: opponent's stones
    channels[1] = (board_arr == opponent).astype(np.float32)

    # Channel 2: metadata
    if active_macro is not None:
        my, mx = active_macro[0], active_macro[1]
        channels[2, my * 3:(my + 1) * 3, mx * 3:(mx + 1) * 3] = 1.0
    else:
        # Free move: mark unresolved macro-boards
        for my in range(3):
            for mx in range(3):
                if macro_arr[my, mx] == 0:
                    channels[2, my * 3:(my + 1) * 3, mx * 3:(mx + 1) * 3] = 1.0 / 9.0

    # Player offset
    offset = 0.25 if current_player == 1 else -0.25
    channels[2] += offset

    return torch.from_numpy(channels)


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------


def load_network(
    checkpoint_path: str, device: str = "cpu"
) -> PolicyValueNetwork:
    """Load a PolicyValueNetwork from a ``.pt`` checkpoint file.

    Supports both plain state dicts and pipeline-style checkpoints
    (which include architecture metadata wrapping a ``"state_dict"`` key).

    The network is restored to *eval* mode and returned on the requested
    device.

    Args:
        checkpoint_path: Path to the saved ``.pt`` checkpoint.
        device: Target device (``'cpu'`` or ``'cuda'``).

    Returns:
        A :class:`PolicyValueNetwork` instance in eval mode.
    """
    raw = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )

    # Detect pipeline-style checkpoint (metadata wrapping)
    if isinstance(raw, dict) and "state_dict" in raw:
        channels = raw.get("channels", 160)
        num_res_blocks = raw.get("num_res_blocks", 10)
        network = PolicyValueNetwork(
            channels=channels, num_res_blocks=num_res_blocks
        )
        network.load_state_dict(raw["state_dict"])
    else:
        # Plain state dict — use default architecture
        network = PolicyValueNetwork()
        network.load_state_dict(raw)

    network.eval()
    return network


def save_network(network: PolicyValueNetwork, path: str) -> None:
    """Save a PolicyValueNetwork's state dict to a ``.pt`` file.

    Args:
        network: The network to save.
        path: Destination path for the ``.pt`` file.
    """
    torch.save(network.state_dict(), path)


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------


def get_masked_policy(
    policy_logits: torch.Tensor, valid_actions: List[List[int]]
) -> torch.Tensor:
    """Mask logits for illegal actions and apply softmax.

    Args:
        policy_logits: A 1-D tensor of shape ``(81,)`` containing raw
            logits for all 81 board cells (index = ``y * 9 + x``).
        valid_actions: A list of ``[x, y]`` legal actions.

    Returns:
        A 1-D tensor of shape ``(81,)`` of probabilities that sum to 1,
        with zeros at positions corresponding to illegal actions.
    """
    # Build mask of shape (81,) — True for legal indices
    mask = torch.zeros(81, dtype=torch.bool)
    for x, y in valid_actions:
        idx = y * 9 + x
        mask[idx] = True

    # Apply mask: set illegal logits to a very large negative number
    masked_logits = policy_logits.clone()
    masked_logits[~mask] = -1e9

    # Softmax over all 81 entries
    probs = F.softmax(masked_logits, dim=0)
    return probs
