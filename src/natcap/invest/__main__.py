import multiprocessing
import sys

# We want to guarantee that this is called BEFORE any other processes start,
# which could happen at import time.
if __name__ == '__main__':
    multiprocessing.freeze_support()

from . import cli

if __name__ == '__main__':
    sys.exit(cli.main())
