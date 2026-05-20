"""Board symmetry transformations for data augmentation.

Ultimate Tic-Tac-Toe is invariant under the dihedral group D4
(4 rotations + 4 reflections).  Augmenting training examples with all 8
symmetries multiplies the effective dataset size and helps the network
learn faster.
"""

from typing import Callable, List, Tuple

import numpy as np


# Type alias for a transformation function
TransformFn = Callable[[np.ndarray], np.ndarray]


def _make_policy_perm(transform_idx: int) -> np.ndarray:
    """Build a permutation array of shape (81,) for the given transform.

    Args:
        transform_idx: Index into ``ALL_TRANSFORMS`` (0-7).

    Returns:
        A length-81 array where ``perm[old_index] = new_index``.
    """
    perm = np.empty(81, dtype=np.int64)
    for y in range(9):
        for x in range(9):
            old_idx = y * 9 + x
            x2, y2 = _coord_transform(x, y, transform_idx)
            new_idx = y2 * 9 + x2
            perm[old_idx] = new_idx
    return perm


def _coord_transform(x: int, y: int, idx: int) -> Tuple[int, int]:
    """Apply the idx-th D4 symmetry to a coordinate on a 9×9 board."""
    if idx == 0:  # identity
        return x, y
    if idx == 1:  # rot 90 CCW
        return y, 8 - x
    if idx == 2:  # rot 180
        return 8 - x, 8 - y
    if idx == 3:  # rot 270 CCW
        return 8 - y, x
    if idx == 4:  # horizontal flip (mirror over vertical axis)
        return 8 - x, y
    if idx == 5:  # vertical flip (mirror over horizontal axis)
        return x, 8 - y
    if idx == 6:  # main diagonal flip (transpose)
        return y, x
    if idx == 7:  # anti-diagonal flip
        return 8 - y, 8 - x
    raise ValueError(f"Invalid transform index {idx}")


# Pre-compute permutations for all 8 transforms
_POLICY_PERMS: List[np.ndarray] = [
    _make_policy_perm(i) for i in range(8)
]


def _board_transforms() -> List[TransformFn]:
    """Return the 8 board-channel transforms."""
    return [
        lambda b: b,  # identity
        lambda b: np.rot90(b, k=1, axes=(1, 2)),
        lambda b: np.rot90(b, k=2, axes=(1, 2)),
        lambda b: np.rot90(b, k=3, axes=(1, 2)),
        lambda b: np.flip(b, axis=2),  # horizontal flip
        lambda b: np.flip(b, axis=1),  # vertical flip
        lambda b: np.transpose(b, (0, 2, 1)),  # transpose (x/y swap)
        lambda b: np.flip(np.transpose(b, (0, 2, 1)), axis=(1, 2)),  # anti-diag
    ]


BOARD_TRANSFORMS: List[TransformFn] = _board_transforms()


def augment_example(
    state_features: np.ndarray,
    search_policy: np.ndarray,
    outcome: float,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generate all 8 D4 symmetries of a single training example.

    Args:
        state_features: Board encoding of shape ``(3, 9, 9)``.
        search_policy: 81-dimensional policy vector.
        outcome: Scalar outcome value.

    Returns:
        A list of 8 ``(state_features', policy', outcome)`` tuples, one
        for each symmetry in the dihedral group D4.
    """
    results: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for idx, transform in enumerate(BOARD_TRANSFORMS):
        aug_board = transform(state_features)
        perm = _POLICY_PERMS[idx]
        # perm[old_index] = new_index, so we need the inverse to map
        # new_index -> old_index for indexing into the original policy.
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        aug_policy = search_policy[inv_perm]
        results.append((aug_board, aug_policy, outcome))
    return results
