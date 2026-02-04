#!/usr/bin/env python
"""
Backward compatibility wrapper.

This script has been moved to scripts/experiments/run_single_model.py.
This wrapper maintains backward compatibility with existing command paths.
"""
from scripts.experiments.run_single_model import main

if __name__ == '__main__':
    main()
