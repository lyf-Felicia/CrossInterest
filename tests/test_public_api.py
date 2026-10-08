import json

import pytest

from crossinterest.cli import configured_command, main
from crossinterest.evaluation.common import latest_details, score_response


def test_cli_doctor(capsys):
    assert main(['doctor']) == 0
    assert 'data_root' in json.loads(capsys.readouterr().out)


def test_distributed_training_command():
    command = configured_command('train', {'command': 'train', 'launcher': {'processes': 4},
        'arguments': {'view': 'behavior_interest', 'gradient-checkpointing': True}})
    assert '--nproc-per-node=4' in command
    assert 'crossinterest.training.train' in command
    assert '--gradient-checkpointing' in command


def test_reject_candidate_evaluation_parallelism():
    with pytest.raises(ValueError, match='one process'):
        configured_command('evaluate', {'command': 'evaluate', 'launcher': {'processes': 4}, 'arguments': {}})


def test_exact_response_scoring():
    result = score_response('New topic: Lighting', ['Lighting', 'Bedding'], 'Lighting')
    assert result['correct']


def test_cached_prediction_identity(tmp_path):
    path = tmp_path / 'details.jsonl'
    path.write_text(json.dumps({'sample_id': 'sample', 'run_fingerprint': 'expected'}) + '\n')
    assert len(latest_details(path, 'expected')) == 1
    with pytest.raises(ValueError, match='fingerprint'):
        latest_details(path, 'wrong')