"""Frozen-baseline smoke test on four single-pill images and one empty image.

Reference counts are used only after inference, never as model input.
No manual bbox ground truth: this is not an mAP or localization-accuracy test.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

CLASSES = ('drug001', 'drug002', 'drug003', 'drug004')
MODEL_SHA256 = 'd195469586c0511b28c5052735372e609d5003c5923c3fb23ec207d22055a0f4'
CASES = {f'v2-{c}.jpg': c for c in CLASSES} | {'v2-空白.jpg': None}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cases(cases):
    if not isinstance(cases, dict) or len(cases) != 5:
        raise ValueError('Cases must map exactly five filenames to four classes and one null')
    for name, reference in cases.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError('Case filenames must be nonempty basenames')
        if reference is not None and reference not in CLASSES:
            raise ValueError('Unknown reference class')
    if list(cases.values()).count(None) != 1 or any(
            list(cases.values()).count(c) != 1 for c in CLASSES):
        raise ValueError('Exactly one case per class and one empty case are required')


def assess(reference, predictions):
    counts = {c: sum(p['class_name'] == c for p in predictions) for c in CLASSES}
    expected = {c: int(c == reference) for c in CLASSES}
    return dict(expected_counts=expected, detected_counts=counts,
                counts_match_reference=counts == expected,
                no_detection_of_present_pill=reference is not None and not predictions,
                expected_class_absent=reference is not None and counts[reference] == 0,
                wrong_class_present=reference is not None and any(
                    p['class_name'] != reference for p in predictions),
                multiple_boxes=len(predictions) > 1,
                background_false_positive=reference is None and bool(predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', type=Path, help='Optional JSON filename-to-reference-class mapping')
    args = parser.parse_args()
    try:
        cases = json.loads(args.cases.read_text()) if args.cases else CASES
        validate_cases(cases)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error('Output exists; use a new run name. Never overwrite results.')
    if sha(args.model) != MODEL_SHA256:
        parser.error('Model does not match frozen baseline')
    files = {p.name for p in args.images.iterdir() if p.is_file() and p.name != '.DS_Store'}
    if files != set(cases):
        parser.error(f'Expected exactly these five files: {list(cases)}; found {sorted(files)}')
    hashes = {name: sha(args.images / name) for name in cases}
    cases_hash = sha(args.cases) if args.cases else None
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    if model.names != dict(enumerate(CLASSES)):
        parser.error('Unexpected class mapping')
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ('annotated', 'labels'):
        (args.output / folder).mkdir()
    settings = dict(imgsz=640, conf=0.25, iou=0.7, device='cpu', max_det=300,
                    rect=True, quantize=None, augment=False, agnostic_nms=False,
                    save=False, verbose=False)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  device_location=f'{platform.system()} {platform.machine()} CPU',
                  execution_hostname=platform.node(),
                  model_path=str(args.model.resolve()), model_sha256=MODEL_SHA256,
                  script_sha256=sha(Path(__file__)), input_sha256=hashes,
                  cases=cases, cases_sha256=cases_hash,
                  python=sys.version, platform=platform.platform(),
                  packages={p: version(p) for p in ('torch', 'torchvision', 'ultralytics', 'Pillow')},
                  settings=settings, images=[],
                  limitations='Five exploratory cases, one per class and one empty image. '
                  'No bbox ground truth, mAP, blind-test claim, or clinical safety inference. '
                  'All cases retained without confidence tuning or training. '
                  'No additional cropping or enhancement inside inference runner; '
                  'inputs may be derived images, see run-specific preprocessing documentation.')
    for name, reference in cases.items():
        record = dict(image=name, reference_class=reference)
        try:
            with Image.open(args.images / name) as raw:
                record.update(raw_size=list(raw.size), exif_orientation=raw.getexif().get(274, 1))
                image = ImageOps.exif_transpose(raw).convert('RGB')
                image.load()
            record['oriented_size'] = list(image.size)
            result = model.predict(source=image, **settings)[0]
            predictions = [dict(class_name=CLASSES[int(b.cls.item())],
                                confidence=float(b.conf.item()), xyxy=b.xyxy[0].tolist())
                           for b in result.boxes]
            record.update(predictions=predictions, stage_ms=result.speed,
                          **assess(reference, predictions))
            result.save(filename=str(args.output / 'annotated' / (Path(name).stem + '.png')))
            # Write an explicit empty label file when no boxes exist.
            lines = []
            for b in result.boxes:
                values = b.xywhn[0].tolist()
                lines.append(' '.join([str(int(b.cls.item()))] +
                                      [f'{v:.9f}' for v in values] + [f'{b.conf.item():.9f}']))
            (args.output / 'labels' / (Path(name).stem + '.txt')).write_text('\n'.join(lines))
            print(name, json.dumps(record['detected_counts']),
                  'COUNT_MATCH' if record['counts_match_reference'] else 'REVIEW_FAILURE', flush=True)
        except Exception as exc:
            record.update(processing_error=repr(exc), counts_match_reference=False)
            print(name, 'PROCESSING_ERROR', repr(exc), flush=True)
        report['images'].append(record)
    if sha(args.model) != MODEL_SHA256 or any(sha(args.images / n) != h for n, h in hashes.items()):
        raise RuntimeError('Input/model integrity changed; results must not be used')
    if args.cases and sha(args.cases) != cases_hash:
        raise RuntimeError('Cases file changed during run; results must not be used')
    report['summary'] = dict(images=len(cases),
                            exact_count_images=sum(r['counts_match_reference'] for r in report['images']),
                            processing_errors=sum('processing_error' in r for r in report['images']))
    (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Saved:', args.output, flush=True)
    return bool(report['summary']['processing_errors'])


if __name__ == '__main__':
    raise SystemExit(main())
