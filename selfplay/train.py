"""
Training utilities for the AlphaZero-lite self-play pipeline.

Provides :func:`prepare_batches` to shuffle and batch training examples,
and :func:`train_network` to perform one iteration of supervised learning
on the collected self-play data.
"""

import logging
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

from engine.policy_value_network import PolicyValueNetwork
from selfplay.config import SelfPlayConfig
from selfplay.self_play import TrainingExample

logger = logging.getLogger(__name__)


def prepare_batches(
    examples: List[TrainingExample],
    batch_size: int,
    shuffle_rng: random.Random,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert a list of training examples into batched tensors.

    The examples are shuffled, grouped into batches of *batch_size*, and
    each batch is converted to a tuple of
    ``(state_tensor, policy_tensor, value_tensor)``.

    Args:
        examples: The training examples from self-play.
        batch_size: Number of examples per batch.
        shuffle_rng: A seeded ``random.Random`` instance for shuffling.

    Returns:
        A list of ``(states, policies, values)`` tuples, where:
          - ``states``: ``(B, 3, 9, 9)`` float32 tensor.
          - ``policies``: ``(B, 81)`` float32 tensor.
          - ``values``: ``(B, 1)`` float32 tensor.
    """
    indices = list(range(len(examples)))
    shuffle_rng.shuffle(indices)

    batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]

        states_list = []
        policies_list = []
        values_list = []

        for idx in batch_indices:
            ex = examples[idx]
            states_list.append(torch.from_numpy(ex.state_features))
            policies_list.append(torch.from_numpy(ex.search_policy))
            values_list.append(torch.tensor([ex.outcome], dtype=torch.float32))

        states = torch.stack(states_list)  # (B, 3, 9, 9)
        policies = torch.stack(policies_list)  # (B, 81)
        values = torch.stack(values_list)  # (B, 1)

        batches.append((states, policies, values))

    return batches


def train_network(
    network: PolicyValueNetwork,
    examples: List[TrainingExample],
    config: SelfPlayConfig,
    device: str = "cpu",
) -> PolicyValueNetwork:
    """Train the network on self-play examples for one iteration.

    Uses a combined loss function::

        L = (v - z)²  -  Σ π · log(softmax(p))  +  c · ||θ||²

    where:
      - ``v`` is the network's value prediction.
      - ``z`` is the game outcome (+1, -1, 0).
      - ``π`` is the MCTS search policy (target).
      - ``p`` is the network's policy logits.
      - ``c`` is the L2 regularisation coefficient (applied via Adam
        ``weight_decay``).

    Args:
        network: The PolicyValueNetwork to train.  This instance is
            modified in place.
        examples: Training examples from self-play.
        config: Self-play configuration (learning rate, batch size, etc.).
        device: Device to use for training (``'cpu'`` or ``'cuda'``).

    Returns:
        The trained ``PolicyValueNetwork`` (same instance, updated weights).
    """
    if not examples:
        logger.warning("No training examples provided; skipping training.")
        return network

    network.train()
    network.to(device)

    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_regularization,
    )

    shuffle_rng = random.Random(42)

    for epoch in range(config.epochs):
        batches = prepare_batches(examples, config.batch_size, shuffle_rng)
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = len(batches)

        for states, target_policies, target_values in batches:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            # Forward pass
            policy_logits, values = network(states)

            # Policy loss: cross-entropy between target policy and softmax(p)
            # -Σ π · log(softmax(p))
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(values.squeeze(1), target_values.squeeze(1))

            # Combined loss
            loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        avg_loss = total_loss / max(1, num_batches)
        avg_pol_loss = total_policy_loss / max(1, num_batches)
        avg_val_loss = total_value_loss / max(1, num_batches)

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"loss={avg_loss:.4f} "
            f"(policy={avg_pol_loss:.4f}, value={avg_val_loss:.4f})"
        )

    network.eval()
    return network
