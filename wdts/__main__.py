"""Allow ``python -m wdts`` to invoke the WDTS command line."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
