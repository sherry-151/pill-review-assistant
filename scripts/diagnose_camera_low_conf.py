"""One prespecified conf=0.001 diagnostic; never a deployment acceptance test."""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from evaluate_camera_minimal import CLASSES, MODEL_SHA256, assess, sha, validate_cases


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def run(args, *, conf, purpose, entrypoint=None, case_validator=validate_cases,
        limitations=None):
    parser = build_parser()
    if args.output.exists():
        parser.error('Preserve previous diagnostic outputs; do not overwrite')
    if sha(args.model) != MODEL_SHA256:
        parser.error('Frozen model hash mismatch')
    cases = json.loads(args.cases.read_text())
    case_validator(cases)
    if {p.name for p in args.images.iterdir() if p.is_file() and p.name != '.DS_Store'} != set(cases):
        parser.error('Expected exactly the manifest images')
    hashes = {n: sha(args.images/n) for n in cases}
    preparation_hash = sha(args.preparation)
    cases_hash = sha(args.cases)
    preparation = json.loads(args.preparation.read_text())
    if preparation['roi_xyxy'] != [1152, 648, 3456, 1944]:
        parser.error('Unexpected ROI')
    if hashes != {r['derived_image']: r['derived_sha256'] for r in preparation['images']}:
        parser.error('Images do not match the previously fixed crops')
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    if model.names != dict(enumerate(CLASSES)):
        parser.error('Unexpected model class mapping')
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ('annotated-all', 'drug004-only-display', 'labels'):
        (args.output/folder).mkdir()
    settings = dict(imgsz=640, conf=conf, iou=0.7, device='cpu', max_det=300,
                    rect=True, quantize=None, augment=False, agnostic_nms=False,
                    save=False, verbose=False)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  purpose=purpose,
                  model_sha256=MODEL_SHA256, script_sha256=sha(Path(__file__)),
                  entrypoint_sha256=sha(entrypoint or Path(__file__)),
                  helper_sha256=sha(Path(__file__).with_name('evaluate_camera_minimal.py')),
                  input_sha256=hashes, cases_sha256=cases_hash,
                  preparation_sha256=preparation_hash, roi_xyxy=preparation['roi_xyxy'],
                  settings=settings, python=sys.version, platform=platform.platform(),
                  execution_hostname=platform.node(),
                  packages={p:version(p) for p in ('torch','torchvision','ultralytics','Pillow')},
                  images=[], limitations=limitations or 'Post-confidence/post-NMS detections, not unfiltered raw features. '
                  'All classes retained in JSON and annotated-all. drug004-only-display is an inspection '
                  'view of identical inference results, not filtered inference or proof of correctness. '
                  'One explicitly selected development configuration per run. '
                  'No training, extra crop, clinical accuracy claim or Pi run. '
                  'These cases have been inspected during development; not independent validation.')
    for name, reference in cases.items():
        row = dict(image=name, reference_class=reference)
        try:
            with Image.open(args.images/name) as raw:
                if raw.size != (2304, 1296):
                    raise ValueError('Unexpected crop size')
                image = ImageOps.exif_transpose(raw).convert('RGB')
                image.load()
            result = model.predict(source=image, **settings)[0]
            predictions = [dict(class_name=CLASSES[int(b.cls.item())],
                                confidence=float(b.conf.item()), xyxy=b.xyxy[0].tolist())
                           for b in result.boxes]
            row.update(predictions=predictions, stage_ms=result.speed,
                       hit_max_det=len(predictions) >= settings['max_det'],
                       **assess(reference,predictions))
            result.save(filename=str(args.output/'annotated-all'/name))
            result[result.boxes.cls == 3].save(filename=str(args.output/'drug004-only-display'/name))
            lines=[]
            for b in result.boxes:
                lines.append(' '.join([str(int(b.cls.item()))] +
                                      [f'{v:.9f}' for v in b.xywhn[0].tolist()] +
                                      [f'{b.conf.item():.9f}']))
            (args.output/'labels'/Path(name).with_suffix('.txt')).write_text('\n'.join(lines))
            print(name, row['detected_counts'],
                  'drug004 scores:', [p['confidence'] for p in predictions if p['class_name']=='drug004'],
                  flush=True)
        except Exception as exc:
            row['processing_error']=repr(exc)
            print(name,'PROCESSING_ERROR',repr(exc),flush=True)
        report['images'].append(row)
    if (sha(args.model) != MODEL_SHA256 or sha(args.cases) != cases_hash or
            sha(args.preparation) != preparation_hash or
            any(sha(args.images/n) != h for n,h in hashes.items())):
        raise RuntimeError('Input integrity changed during diagnosis')
    report['summary']=dict(images=len(cases),
                           processing_errors=sum('processing_error' in r for r in report['images']),
                           max_det_hit_images=sum(r.get('hit_max_det',False) for r in report['images']),
                           exact_count_images=sum(r.get('counts_match_reference',False) for r in report['images']))
    (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Saved:',args.output)
    return bool(report['summary']['processing_errors'])


def main():
    return run(build_parser().parse_args(), conf=0.001,
               purpose='DIAGNOSTIC_ONLY_NOT_DEMONSTRATION_PASS')


if __name__=='__main__':
    raise SystemExit(main())
