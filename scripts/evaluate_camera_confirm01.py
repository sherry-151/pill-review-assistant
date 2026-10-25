"""One frozen 15-image confirmation run: fixed central ROI, conf .10, CPU, imgsz 640."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from diagnose_camera_low_conf import run as infer
from evaluate_camera_minimal import CLASSES, MODEL_SHA256, sha
from prepare_camera_roi import run as prepare


def expected_cases(suffix):
    if suffix not in ('.jpg', '.png'):
        raise ValueError('Only original JPG or derived PNG manifests are supported')
    return {f'v3c01_{label}_{i:03d}{suffix}': None if label == 'empty' else label
            for label in (*CLASSES, 'empty') for i in range(1, 4)}


def validate_confirmation_cases(cases):
    if not isinstance(cases, dict) or not cases:
        raise ValueError('Expected the complete 15-image confirmation manifest')
    suffix = Path(next(iter(cases))).suffix
    if cases != expected_cases(suffix):
        raise ValueError('Exactly three correctly named cases per class and three empty cases required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Unmodified original JPG folder')
    parser.add_argument('--derived', type=Path, required=True, help='New lossless crop folder')
    parser.add_argument('--output', type=Path, required=True, help='New run record folder')
    args = parser.parse_args()
    if args.output.exists() or args.derived.exists():
        parser.error('Never overwrite an existing confirmation run or derived folder')
    if sha(args.model) != MODEL_SHA256:
        parser.error('Frozen model hash mismatch')
    cases = expected_cases('.jpg')
    files = {p.name for p in args.images.iterdir() if p.is_file() and p.name != '.DS_Store'}
    if files != set(cases):
        parser.error('Expected exactly the 15 prescribed original JPGs')
    args.output.mkdir(parents=True, exist_ok=False)
    source_cases = args.output/'source-cases.json'
    derived_cases = args.output/'derived-cases.json'
    preparation = args.output/'preparation.json'
    source_cases.write_text(json.dumps(cases, indent=2)+'\n')
    derived_cases.write_text(json.dumps(expected_cases('.png'), indent=2)+'\n')
    prepare(SimpleNamespace(images=args.images, cases=source_cases,
                            output=args.derived, record=preparation),
            case_validator=validate_confirmation_cases)
    return infer(SimpleNamespace(model=args.model, images=args.derived, cases=derived_cases,
                                 preparation=preparation, output=args.output/'inference'),
                 conf=0.10, purpose='FROZEN_CONFIGURATION_NEW_PHOTO_INTERNAL_CONFIRMATION',
                 entrypoint=Path(__file__), case_validator=validate_confirmation_cases,
                 limitations='15 new photos evaluated once with previously selected frozen settings. '
                 'Small internal same-setup confirmation, not clinical or cross-site validation. '
                 'All post-confidence/post-NMS classes and failures retained. '
                 'Reference counts used for assessment only, never as model input. '
                 'No manual bbox ground truth, mAP or localization IoU. '
                 'No confidence tuning, retraining, alternate ROI or Pi execution in this run.')


if __name__ == '__main__':
    raise SystemExit(main())
