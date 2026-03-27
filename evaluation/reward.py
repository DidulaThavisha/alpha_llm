"""Reward computation for code generation games.

Provides partial credit to give gradient signal even from failed attempts.
"""

from .code_executor import ExecutionResult


def compute_reward(result: ExecutionResult) -> float:
    """Compute reward from execution result.

    Returns:
        +1.0           : all tests pass (win)
        -0.5 to ~0.5   : some tests pass (partial credit)
        -0.5           : runs but all tests fail
        -1.0           : crash/timeout (no output at all)
    """
    if result.total == 0:
        return -1.0

    # Check if code crashed on every test case
    num_crashes = sum(1 for o in result.outputs if o is None)
    if num_crashes == result.total:
        return -1.0

    ratio = result.passed / result.total

    if ratio == 1.0:
        return 1.0  # full win
    elif ratio > 0:
        return -0.5 + ratio  # partial credit: range [-0.5, 0.5)
    else:
        return -0.5  # runs but all wrong
