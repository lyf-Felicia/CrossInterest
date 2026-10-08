import importlib.util
from pathlib import Path

import pytest
import yaml

from crossinterest.cli import configured_command


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('action', ['train', 'evaluate'])
def test_public_configurations(action):
    directory = 'training' if action == 'train' else 'evaluation'
    config = yaml.safe_load((ROOT / 'configs' / directory / 'home112.yaml').read_text())
    command = configured_command(action, config)
    assert config['arguments']['view'] == 'behavior_interest'
    assert config['arguments']['label-mode'] == 'interest_id'
    expected = 'crossinterest.training.train' if action == 'train' else 'crossinterest.training.evaluate_ranking'
    assert expected in command


def test_ranking_protocol_is_not_candidate_constrained():
    config = yaml.safe_load((ROOT / 'configs/evaluation/home112.yaml').read_text())
    assert config['arguments']['decoding'] == 'direct_beam_search'
    assert config['arguments']['num-beams'] == 20
    assert config['arguments']['eval-batch-size'] == 1
    assert config['arguments']['label-file'] == 'data/amazon/home112/train.parquet'


def test_repository_publication_boundary():
    spec = importlib.util.spec_from_file_location('repository_checks', ROOT / 'tools/check_repository.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.check()
    assert not result['errors'], '\n'.join(result['errors'])


def test_duplicate_public_content_is_rejected(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('duplicate_checks', ROOT / 'tools/check_repository.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/README.md').write_text('Dataset placement instructions.\n')
    (tmp_path / 'original.md').write_text('A research note.\n')
    (tmp_path / 'redundant.md').write_text('A research note.\n')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    assert any('Duplicate public content:' in error for error in module.check()['errors'])


def test_private_workspace_is_rejected(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('standalone_checks', ROOT / 'tools/check_repository.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/README.md').write_text('Dataset placement instructions.\n')
    (tmp_path / '.local').mkdir()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    assert any('Private workspace embedded' in error for error in module.check()['errors'])


def test_installed_package_does_not_select_environment_parent_checkout(tmp_path, monkeypatch):
    from crossinterest import paths

    current = tmp_path / 'current'
    unrelated = tmp_path / 'unrelated'
    for checkout in (current, unrelated):
        (checkout / 'src/crossinterest').mkdir(parents=True)
        (checkout / 'pyproject.toml').touch()
    installed = unrelated / '.venv/lib/python3.10/site-packages/crossinterest/paths.py'
    monkeypatch.setattr(paths, '__file__', str(installed))
    monkeypatch.chdir(current)
    monkeypatch.delenv('CROSSINTEREST_PROJECT_ROOT', raising=False)
    assert paths.project_root() == current