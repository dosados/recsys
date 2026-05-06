import logging
import sys
from typing import Optional


def setup_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    log_level: Optional[str] = None,
) -> None:
    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    elif quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(handler)

    # third-party noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
