import os
from pathlib import Path


def project_root():
    configured = os.environ.get('CROSSINTEREST_PROJECT_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    current = Path.cwd().resolve()
    for candidate in (source_root, current, *current.parents):
        if (candidate / 'pyproject.toml').is_file() and (candidate / 'src/crossinterest').is_dir():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = project_root()
DATA_ROOT = Path(os.environ.get('CROSSINTEREST_DATA_ROOT', PROJECT_ROOT / 'data')).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get('CROSSINTEREST_OUTPUT_ROOT', PROJECT_ROOT / 'outputs')).expanduser().resolve()


def dataset_dir(name='home112'):
    return DATA_ROOT / 'amazon' / name


def experiment_dir(name):
    return OUTPUT_ROOT / 'experiments' / name


def source_workspace(package):
    legacy = os.environ.get('CROSSINTEREST_LEGACY_WORKSPACE')
    if legacy:
        base = Path(legacy).expanduser().resolve() / 'crossinterest_formal'
        return base / package if package == 'superloss_experiment' else base / 'amazon' / package
    names = {
        'prep': 'dataset', 'eval_amazon_notrain': 'zero_shot',
        'eval_amazon_notrain_112k': 'zero_shot_home112',
        'eval_amazon_sft_112': 'supervised_learning',
        'eval_amazon_sft_99': 'supervised_learning_99',
        'eval_amazon_datasize': 'data_scaling', 'eval_amazon_datasize_99': 'data_scaling_99',
        'eval_amazon_denoise': 'denoising', 'eval_amazon_llm_summary': 'summaries',
        'eval_baseline_google': 'google_baseline', 'eval_baseline_mera': 'mera',
        'mera_qwen2_official_strict': 'mera_strict',
        'eval_interest_id_minionerec': 'alignment', 'superloss_experiment': 'superloss',
    }
    return experiment_dir(names[package])