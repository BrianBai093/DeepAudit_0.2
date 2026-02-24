"""Simple calculator functions with a command-line interface."""

from __future__ import annotations

import argparse
from typing import Callable


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("division by zero")
    return a / b


def main() -> None:
    parser = argparse.ArgumentParser(description="Perform a simple arithmetic operation.")
    parser.add_argument("op", choices=["add", "sub", "mul", "div"], help="Operation to perform")
    parser.add_argument("a", type=float, help="First operand")
    parser.add_argument("b", type=float, help="Second operand")
    args = parser.parse_args()

    operations: dict[str, Callable[[float, float], float]] = {
        "add": add,
        "sub": subtract,
        "mul": multiply,
        "div": divide,
    }

    result = operations[args.op](args.a, args.b)
    print(result)


if __name__ == "__main__":
    main()
