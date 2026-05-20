"""Tests for engine.symmetry board augmentation."""

import numpy as np
import pytest

from engine.symmetry import augment_example


class TestAugmentExample:
    """Tests for augment_example generating 8 D4 symmetries."""

    def test_returns_eight_examples(self) -> None:
        """augment_example returns exactly 8 examples."""
        board = np.zeros((3, 9, 9), dtype=np.float32)
        policy = np.zeros(81, dtype=np.float32)
        policy[0] = 1.0
        examples = augment_example(board, policy, 1.0)
        assert len(examples) == 8

    def test_outcomes_unchanged(self) -> None:
        """The outcome is preserved across all symmetries."""
        board = np.zeros((3, 9, 9), dtype=np.float32)
        policy = np.zeros(81, dtype=np.float32)
        examples = augment_example(board, policy, -0.5)
        for _, _, outcome in examples:
            assert outcome == pytest.approx(-0.5)

    def test_identity_first(self) -> None:
        """The first augmented example is the identity transform."""
        board = np.arange(3 * 9 * 9, dtype=np.float32).reshape(3, 9, 9)
        policy = np.arange(81, dtype=np.float32)
        examples = augment_example(board, policy, 0.0)
        aug_board, aug_policy, aug_outcome = examples[0]
        assert np.array_equal(aug_board, board)
        assert np.array_equal(aug_policy, policy)
        assert aug_outcome == 0.0

    def test_rot90_moves_cells(self) -> None:
        """A 90° rotation moves the top-left corner to bottom-left."""
        board = np.zeros((3, 9, 9), dtype=np.float32)
        board[0, 0, 0] = 1.0  # top-left (y=0, x=0)
        policy = np.zeros(81, dtype=np.float32)
        policy[0] = 1.0  # (0,0) -> index 0
        examples = augment_example(board, policy, 0.0)
        # Index 1 in BOARD_TRANSFORMS is rot90 CCW (axes 1,2)
        rot_board, rot_policy, _ = examples[1]
        # (y=0, x=0) rotated 90 CCW becomes (y=8, x=0)
        assert rot_board[0, 8, 0] == pytest.approx(1.0)
        # Policy index for (x=0, y=8) is 8*9+0 = 72
        assert rot_policy[72] == pytest.approx(1.0)

    def test_policies_sum_to_one(self) -> None:
        """All augmented policies still sum to 1."""
        board = np.zeros((3, 9, 9), dtype=np.float32)
        policy = np.full(81, 1.0 / 81, dtype=np.float32)
        examples = augment_example(board, policy, 0.0)
        for _, aug_policy, _ in examples:
            assert aug_policy.sum() == pytest.approx(1.0, abs=1e-5)

    def test_board_shape_preserved(self) -> None:
        """All augmented boards keep shape (3, 9, 9)."""
        board = np.zeros((3, 9, 9), dtype=np.float32)
        policy = np.zeros(81, dtype=np.float32)
        examples = augment_example(board, policy, 0.0)
        for aug_board, _, _ in examples:
            assert aug_board.shape == (3, 9, 9)
            assert aug_board.dtype == np.float32

    def test_policy_shape_preserved(self) -> None:
        """All augmented policies keep shape (81,)."""
        board = np.zeros((3, 9, 9), dtype=np.float32)
        policy = np.zeros(81, dtype=np.float32)
        examples = augment_example(board, policy, 0.0)
        for _, aug_policy, _ in examples:
            assert aug_policy.shape == (81,)
            assert aug_policy.dtype == np.float32
