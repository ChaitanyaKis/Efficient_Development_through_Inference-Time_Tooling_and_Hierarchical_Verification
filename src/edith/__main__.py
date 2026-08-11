"""Allow ``python -m edith`` as an alternative to the ``edith`` console script."""

from __future__ import annotations

import sys

from edith.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
