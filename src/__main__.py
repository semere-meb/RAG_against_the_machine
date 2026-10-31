import sys

import fire

from src.cli import CLI


def run() -> None:
    try:
        fire.Fire(CLI)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    run()
