"""
Tests for engine.policy_value_network — PolicyValueNetwork, encode_state,
load/save, and get_masked_policy.
"""

import os
import tempfile
import time
from typing import Any, List

import numpy as np
import pytest
import torch

from engine.game_state import UTTTState
from engine.policy_value_network import (
    PolicyValueNetwork,
    ResNetBlock,
    encode_state,
    get_masked_policy,
    load_network,
    save_network,
)


# ---------------------------------------------------------------------------
# ResNetBlock
# ---------------------------------------------------------------------------


class TestResNetBlock:
    """Tests for the ResNetBlock module."""

    def test_constructs(self) -> None:
        """ResNetBlock can be constructed with default channels."""
        block = ResNetBlock()
        assert isinstance(block, ResNetBlock)

    def test_construct_custom_channels(self) -> None:
        """ResNetBlock can be constructed with custom channels."""
        block = ResNetBlock(channels=64)
        assert block.conv1.in_channels == 64
        assert block.conv1.out_channels == 64

    def test_forward_shape(self) -> None:
        """Forward pass preserves spatial dimensions."""
        block = ResNetBlock(channels=32)
        x = torch.randn(4, 32, 9, 9)
        out = block(x)
        assert out.shape == (4, 32, 9, 9)

    def test_forward_values(self) -> None:
        """Forward pass produces finite values (not NaN / inf)."""
        block = ResNetBlock(channels=16)
        x = torch.randn(2, 16, 9, 9)
        out = block(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# PolicyValueNetwork construction
# ---------------------------------------------------------------------------


class TestPolicyValueNetworkConstruction:
    """Tests for PolicyValueNetwork construction and parameter count."""

    def test_default_construction(self) -> None:
        """Network constructs with default arguments."""
        net = PolicyValueNetwork()
        assert isinstance(net, PolicyValueNetwork)
        assert net.channels == 160
        assert net.num_res_blocks == 10

    def test_custom_construction(self) -> None:
        """Network constructs with custom arguments."""
        net = PolicyValueNetwork(channels=64, num_res_blocks=5)
        assert net.channels == 64
        assert len(net.res_blocks) == 5
        assert net.num_res_blocks == 5

    def test_res_block_count(self) -> None:
        """Network has exactly num_res_blocks residual blocks."""
        for n in [1, 5, 10, 20]:
            net = PolicyValueNetwork(num_res_blocks=n)
            assert len(net.res_blocks) == n

    def test_parameter_count_less_than_6m(self) -> None:
        """Default network has <6 million parameters."""
        net = PolicyValueNetwork()
        total = sum(p.numel() for p in net.parameters())
        assert total < 6_000_000, f"Too many params: {total}"

    def test_parameter_count_greater_than_4m(self) -> None:
        """Default network has >4 million parameters."""
        net = PolicyValueNetwork()
        total = sum(p.numel() for p in net.parameters())
        assert total > 4_000_000, f"Too few params: {total}"

    def test_parameter_count_report(self) -> None:
        """Print actual parameter count for reference."""
        net = PolicyValueNetwork()
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(
            f"PolicyValueNetwork params: {total:,} total, "
            f"{trainable:,} trainable"
        )
        # Just ensure it runs; no assertion needed


# ---------------------------------------------------------------------------
# PolicyValueNetwork forward pass
# ---------------------------------------------------------------------------


class TestPolicyValueNetworkForward:
    """Tests for forward pass of PolicyValueNetwork."""

    def test_forward_zero_state(self) -> None:
        """Forward pass on zero input produces correct output shapes."""
        net = PolicyValueNetwork()
        net.eval()
        x = torch.zeros(1, 3, 9, 9)  # batch of 1
        with torch.no_grad():
            policy_logits, value = net(x)
        assert policy_logits.shape == (1, 81), (
            f"Expected (1, 81), got {policy_logits.shape}"
        )
        assert value.shape == (1, 1), (
            f"Expected (1, 1), got {value.shape}"
        )

    def test_forward_batch(self) -> None:
        """Forward pass on batch of 4 produces correct shapes."""
        net = PolicyValueNetwork()
        net.eval()
        x = torch.randn(4, 3, 9, 9)
        with torch.no_grad():
            policy_logits, value = net(x)
        assert policy_logits.shape == (4, 81)
        assert value.shape == (4, 1)

    def test_forward_value_in_range(self) -> None:
        """Value output is in [-1, 1] after tanh."""
        net = PolicyValueNetwork()
        net.eval()
        # Test with many random inputs
        for _ in range(10):
            x = torch.randn(2, 3, 9, 9) * 2.0  # larger variation
            with torch.no_grad():
                _, value = net(x)
            assert value.min() >= -1.0, "Value < -1.0"
            assert value.max() <= 1.0, "Value > 1.0"

    def test_forward_policy_logits_finite(self) -> None:
        """Policy logits are all finite (no NaN / inf)."""
        net = PolicyValueNetwork()
        net.eval()
        x = torch.randn(2, 3, 9, 9)
        with torch.no_grad():
            policy_logits, _ = net(x)
        assert torch.isfinite(policy_logits).all()

    def test_forward_deterministic(self) -> None:
        """Forward pass is deterministic with fixed weights."""
        net = PolicyValueNetwork()
        net.eval()
        x = torch.randn(1, 3, 9, 9)
        with torch.no_grad():
            out1, val1 = net(x)
            out2, val2 = net(x)
        assert torch.equal(out1, out2)
        assert torch.equal(val1, val2)

    def test_inference_time(self) -> None:
        """Benchmark: measure inference time for a single forward pass."""
        net = PolicyValueNetwork()
        net.eval()
        x = torch.randn(1, 3, 9, 9)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                net(x)

        # Timed runs
        n_runs = 100
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                net(x)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / n_runs) * 1000
        print(f"Average inference time: {avg_ms:.4f} ms ({n_runs} runs)")
        # Just benchmark — no strict assertion, but log for visibility
        assert avg_ms < 100.0, (
            f"Inference too slow: {avg_ms:.2f}ms (expected <100ms)"
        )


# ---------------------------------------------------------------------------
# encode_state
# ---------------------------------------------------------------------------


class TestEncodeState:
    """Tests for the encode_state function."""

    def test_returns_tensor(self) -> None:
        """encode_state returns a torch.Tensor."""
        state = UTTTState()
        encoded = encode_state(state)
        assert isinstance(encoded, torch.Tensor)

    def test_shape(self) -> None:
        """encode_state returns a (3, 9, 9) tensor."""
        state = UTTTState()
        encoded = encode_state(state)
        assert encoded.shape == (3, 9, 9), (
            f"Expected (3, 9, 9), got {encoded.shape}"
        )

    def test_dtype(self) -> None:
        """encode_state returns float32 tensor."""
        state = UTTTState()
        encoded = encode_state(state)
        assert encoded.dtype == torch.float32

    def test_empty_board_channels(self) -> None:
        """Empty board: channels 0 and 1 are all zeros."""
        state = UTTTState()
        encoded = encode_state(state)
        assert (encoded[0] == 0).all(), "Channel 0 should be all zeros"
        assert (encoded[1] == 0).all(), "Channel 1 should be all zeros"

    def test_current_player_stones(self) -> None:
        """Channel 0 has 1.0 where current player occupies."""
        board = [[0] * 9 for _ in range(9)]
        board[4][4] = 1  # P1 stone
        state = UTTTState(board=board, current_player=1)
        encoded = encode_state(state)
        assert encoded[0, 4, 4] == 1.0, "P1 stone at (4,4) not detected"
        assert encoded[1, 4, 4] == 0.0, (
            "Opponent channel should be 0 at P1 stone"
        )

    def test_opponent_stones(self) -> None:
        """Channel 1 has 1.0 where opponent occupies."""
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = 2  # P2 stone
        state = UTTTState(board=board, current_player=1)
        encoded = encode_state(state)
        assert encoded[0, 0, 0] == 0.0, "Channel 0 should be 0 for P2 stone"
        assert encoded[1, 0, 0] == 1.0, "P2 stone at (0,0) not detected"

    def test_both_players_stones(self) -> None:
        """Both channels correctly encode respective player stones."""
        board = [[0] * 9 for _ in range(9)]
        board[0][0] = 1  # P1
        board[1][1] = 2  # P2
        state = UTTTState(board=board, current_player=1)
        encoded = encode_state(state)
        assert encoded[0, 0, 0] == 1.0  # P1
        assert encoded[0, 1, 1] == 0.0  # not P1
        assert encoded[1, 0, 0] == 0.0  # not P2
        assert encoded[1, 1, 1] == 1.0  # P2

    def test_active_macro_region(self) -> None:
        """Channel 2 has 1.0 in active macro-board region."""
        state = UTTTState(active_macro=[1, 1])  # center macro-board
        encoded = encode_state(state)
        # Center macro = rows 3..5, cols 3..5
        for y in range(3, 6):
            for x in range(3, 6):
                assert encoded[2, y, x] == pytest.approx(
                    1.0 + 0.25
                ), f"Expected 1.25 at ({y},{x})"
        # Outside region should be just the offset
        assert encoded[2, 0, 0] == pytest.approx(0.25)

    def test_free_move_uniform(self) -> None:
        """Free move (active_macro=None) gives 1/9 in unresolved boards."""
        state = UTTTState(active_macro=None)  # free move
        encoded = encode_state(state)
        # All boards unresolved -> each cell gets 1/9 + 0.25
        expected = 1.0 / 9.0 + 0.25
        for y in range(9):
            for x in range(9):
                assert encoded[2, y, x] == pytest.approx(
                    expected
                ), f"Mismatch at ({y},{x})"

    def test_free_move_with_resolved_boards(self) -> None:
        """Free move with resolved boards: resolved boards get 0 in meta."""
        macro_board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]  # (0,0) resolved
        state = UTTTState(macro_board=macro_board, active_macro=None)
        encoded = encode_state(state)
        # (0,0) macro board is resolved -> should NOT get 1/9
        for y in range(3):
            for x in range(3):
                assert encoded[2, y, x] == pytest.approx(
                    0.25
                ), f"Resolved board cell ({y},{x}) should be offset only"
        # Other boards are unresolved -> get 1/9
        expected = 1.0 / 9.0 + 0.25
        # Board (0,1): rows 0..2, cols 3..5
        assert encoded[2, 0, 3] == pytest.approx(expected)

    def test_player_offset_p1(self) -> None:
        """Channel 2 has +0.25 offset for current_player == 1."""
        state = UTTTState(current_player=1)
        encoded = encode_state(state)
        # All entries have offset 0.25 (no active macro, all unresolved)
        expected = 1.0 / 9.0 + 0.25
        assert encoded[2, 0, 0] == pytest.approx(expected)

    def test_player_offset_p2(self) -> None:
        """Channel 2 has -0.25 offset for current_player == 2."""
        state = UTTTState(current_player=2)
        encoded = encode_state(state)
        expected = 1.0 / 9.0 - 0.25
        assert encoded[2, 0, 0] == pytest.approx(expected)

    def test_non_constant_active_macro(self) -> None:
        """Active macro + current_player offset compose correctly."""
        state = UTTTState(
            active_macro=[0, 2], current_player=2  # top-right macro
        )
        encoded = encode_state(state)
        # Top-right macro = rows 0..2, cols 6..8: 1.0 + (-0.25) = 0.75
        assert encoded[2, 0, 6] == pytest.approx(0.75)
        # Outside: just offset
        assert encoded[2, 4, 4] == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# get_masked_policy
# ---------------------------------------------------------------------------


class TestGetMaskedPolicy:
    """Tests for the get_masked_policy function."""

    def test_returns_tensor(self) -> None:
        """get_masked_policy returns a torch.Tensor."""
        logits = torch.randn(81)
        valid = [[0, 0], [4, 4]]
        result = get_masked_policy(logits, valid)
        assert isinstance(result, torch.Tensor)

    def test_shape(self) -> None:
        """get_masked_policy returns (81,) tensor."""
        logits = torch.randn(81)
        valid = [[0, 0], [4, 4]]
        result = get_masked_policy(logits, valid)
        assert result.shape == (81,)

    def test_sums_to_one(self) -> None:
        """Probability distribution sums to 1.0."""
        logits = torch.randn(81)
        valid = [[0, 0], [4, 4], [8, 8]]
        result = get_masked_policy(logits, valid)
        assert result.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_zeros_for_illegal_actions(self) -> None:
        """Illegal action indices have probability 0."""
        logits = torch.randn(81)
        valid = [[0, 0], [4, 4]]  # only these two are legal
        result = get_masked_policy(logits, valid)
        for i in range(81):
            x, y = i % 9, i // 9
            if [x, y] not in valid:
                assert result[i].item() == 0.0, (
                    f"Index {i} should be 0 (illegal action)"
                )

    def test_legal_actions_positive(self) -> None:
        """Legal action indices have positive probability."""
        logits = torch.randn(81)
        valid = [[0, 0], [4, 4], [8, 8]]
        result = get_masked_policy(logits, valid)
        for x, y in valid:
            idx = y * 9 + x
            assert result[idx].item() > 0.0, (
                f"Legal action ({x},{y}) should have positive prob"
            )

    def test_single_legal_action(self) -> None:
        """Only one legal action: probability is 1.0 at that index."""
        logits = torch.randn(81)
        valid = [[4, 4]]
        result = get_masked_policy(logits, valid)
        idx = 4 * 9 + 4  # = 40
        assert result[idx].item() == pytest.approx(1.0, abs=1e-6)
        assert result.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_all_actions_legal(self) -> None:
        """All 81 actions legal: distribution sums to 1, all positive."""
        logits = torch.randn(81)
        valid = [[x, y] for y in range(9) for x in range(9)]
        result = get_masked_policy(logits, valid)
        assert result.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert (result > 0).all()

    def test_very_negative_logit_zero_prob(self) -> None:
        """A legal action with very negative logit gets near-zero prob."""
        logits = torch.randn(81)
        logits[40] = -100.0  # (4,4) gets very negative
        valid = [[0, 0], [4, 4], [8, 8]]
        result = get_masked_policy(logits, valid)
        idx_44 = 4 * 9 + 4
        assert result[idx_44].item() > 0.0  # still slightly positive
        # But should be smaller than other actions
        idx_00 = 0
        assert result[40] < result[0]

    def test_deterministic(self) -> None:
        """Same inputs produce same output."""
        logits = torch.tensor(
            [float(i) for i in range(81)], dtype=torch.float32
        )
        valid = [[0, 0], [4, 4], [8, 8]]
        r1 = get_masked_policy(logits, valid)
        r2 = get_masked_policy(logits, valid)
        assert torch.equal(r1, r2)


# ---------------------------------------------------------------------------
# save_network / load_network
# ---------------------------------------------------------------------------


class TestSaveLoadNetwork:
    """Tests for save_network and load_network."""

    def test_save_and_load(self) -> None:
        """Save then load restores network to eval mode with same weights."""
        net = PolicyValueNetwork()
        # Get a reference output for a fixed input
        x = torch.randn(1, 3, 9, 9)
        net.eval()
        with torch.no_grad():
            ref_logits, ref_value = net(x)

        # Save to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            save_network(net, tmp_path)

            # Load
            loaded = load_network(tmp_path)
            assert loaded.training is False, "Network should be in eval mode"

            # Compare outputs
            with torch.no_grad():
                logits, value = loaded(x)
            assert torch.equal(logits, ref_logits), (
                "Policy logits differ after load"
            )
            assert torch.equal(value, ref_value), (
                "Value differs after load"
            )
        finally:
            os.unlink(tmp_path)

    def test_load_eval_mode(self) -> None:
        """load_network returns network in eval mode."""
        net = PolicyValueNetwork()
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            save_network(net, tmp_path)
            loaded = load_network(tmp_path)
            assert not loaded.training
        finally:
            os.unlink(tmp_path)

    def test_save_and_load_different_instance(self) -> None:
        """Loaded network is a different instance from saved one."""
        net = PolicyValueNetwork()
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            save_network(net, tmp_path)
            loaded = load_network(tmp_path)
            assert loaded is not net
        finally:
            os.unlink(tmp_path)

    def test_load_device_param(self) -> None:
        """load_network accepts device parameter."""
        net = PolicyValueNetwork()
        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            save_network(net, tmp_path)
            loaded = load_network(tmp_path, device="cpu")
            assert loaded is not None
        finally:
            os.unlink(tmp_path)

    def test_load_network_with_pipeline_checkpoint(self, tmp_path: Any) -> None:
        """load_network() must handle pipeline-style checkpoints with metadata.

        The self-play pipeline's ``save_checkpoint()`` saves a dict with
        ``state_dict``, ``channels``, and ``num_res_blocks`` keys.  The agent's
        ``load_network()`` must correctly reconstruct the network from this
        format so that checkpoints produced by the pipeline are usable by the
        agent.
        """
        from engine.policy_value_network import (
            PolicyValueNetwork,
            load_network,
        )

        original = PolicyValueNetwork(channels=64, num_res_blocks=2)
        original.eval()

        # Save in pipeline format (metadata-wrapped dict)
        path = str(tmp_path / "pipeline_checkpoint.pt")
        checkpoint = {
            "state_dict": original.state_dict(),
            "channels": 64,
            "num_res_blocks": 2,
        }
        torch.save(checkpoint, path)

        # Load using the agent's load_network()
        loaded = load_network(path)

        # Verify correct architecture
        assert loaded.channels == 64
        assert loaded.num_res_blocks == 2

        # Verify same forward pass output
        dummy_input = torch.randn(1, 3, 9, 9)
        with torch.no_grad():
            orig_out = original(dummy_input)
            loaded_out = loaded(dummy_input)

        assert torch.equal(orig_out[0], loaded_out[0]), (
            "Policy logits differ after loading pipeline checkpoint"
        )
        assert torch.equal(orig_out[1], loaded_out[1]), (
            "Value differs after loading pipeline checkpoint"
        )

    def test_load_network_with_plain_checkpoint(self, tmp_path: Any) -> None:
        """load_network() must still handle plain state dicts."""
        from engine.policy_value_network import (
            PolicyValueNetwork,
            load_network,
        )

        original = PolicyValueNetwork()
        original.eval()
        path = str(tmp_path / "plain_checkpoint.pt")
        torch.save(original.state_dict(), path)

        loaded = load_network(path)

        dummy_input = torch.randn(1, 3, 9, 9)
        with torch.no_grad():
            orig_out = original(dummy_input)
            loaded_out = loaded(dummy_input)

        assert torch.equal(orig_out[0], loaded_out[0]), (
            "Policy logits differ after loading plain checkpoint"
        )
        assert torch.equal(orig_out[1], loaded_out[1]), (
            "Value differs after loading plain checkpoint"
        )
