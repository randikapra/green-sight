#!/usr/bin/env python3
"""Preprocess satellite imagery for all study areas."""

import argparse
import sys
sys.path.insert(0, '.')

from src.utils.config import load_config, setup_environment
from src.utils.logger import get_logger
from src.data.preprocess import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_environment(cfg)
    logger = get_logger('preprocess', cfg.project.log_dir)
    preprocess(cfg, logger)


if __name__ == '__main__':
    main()
