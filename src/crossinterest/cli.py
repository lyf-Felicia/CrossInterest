import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from crossinterest.paths import DATA_ROOT, OUTPUT_ROOT, PROJECT_ROOT


MODULES = {'train': 'crossinterest.training.train', 'evaluate': 'crossinterest.training.evaluate_ranking'}
PATH_OPTIONS = {'train-file', 'label-map', 'output-dir', 'adapter', 'data-dir', 'label-file', 'run-dir', 'init-adapter'}


def configured_command(action, config):
    if config.get('command') != action:
        raise ValueError(f'Configuration command must be {action!r}')
    if set(config) - {'command', 'launcher', 'arguments'}:
        raise ValueError('Unsupported top-level configuration fields')
    arguments = config.get('arguments')
    if not isinstance(arguments, dict):
        raise ValueError('Configuration arguments must be a mapping')
    processes = config.get('launcher', {}).get('processes', 1)
    if not isinstance(processes, int) or processes < 1:
        raise ValueError('launcher.processes must be a positive integer')
    if action != 'train' and processes != 1:
        raise ValueError('Ranking uses one process; device-map may distribute the model')
    command = [sys.executable, '-m', MODULES[action]]
    if processes > 1:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc-per-node={processes}', '-m', MODULES[action]]
    for key, value in arguments.items():
        if not isinstance(key, str) or not key or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for character in key):
            raise ValueError(f'Invalid argument name: {key!r}')
        if isinstance(value, bool):
            command.append('--' + ('' if value else 'no-') + key)
        elif isinstance(value, (str, int, float)):
            if key in PATH_OPTIONS:
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    value = str(PROJECT_ROOT / path)
            command.extend(['--' + key, str(value)])
        else:
            raise ValueError(f'Invalid argument value for {key!r}')
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description='CrossInterest training and candidate-free ranking')
    commands = parser.add_subparsers(dest='action', required=True)
    for action in MODULES:
        command = commands.add_parser(action)
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--dry-run', action='store_true')
    commands.add_parser('doctor', help='Print resolved data and output locations; no model loading')
    args = parser.parse_args(argv)
    if args.action == 'doctor':
        print(json.dumps({'project_root': str(PROJECT_ROOT), 'data_root': str(DATA_ROOT),
                          'output_root': str(OUTPUT_ROOT), 'home112_available': (DATA_ROOT / 'amazon/home112/train.parquet').exists()}, indent=2))
        return 0
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        parser.error('Configuration must be a YAML mapping')
    try:
        command = configured_command(args.action, config)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())