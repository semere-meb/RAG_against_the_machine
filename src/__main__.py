import sys

import fire

from src.cli import CLI


def run() -> None:
    """Run the CLI, reporting errors as messages instead of tracebacks."""
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
