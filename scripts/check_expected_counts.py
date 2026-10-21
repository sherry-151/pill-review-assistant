"""Fixed-model count comparison demo. No training, original edits, or clinical release."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_counts(counts):
    if set(counts) != set(CLASSES):
        raise ValueError('Counts must explicitly include exactly drug001 through drug004')
    if any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('Counts must be nonnegative integers')


def compare_counts(expected, detected):
    validate_counts(expected)
    validate_counts(detected)
    delta = {c: detected[c] - expected[c] for c in CLASSES}
    return dict(delta=delta, status='COUNT_MATCH_REVIEW_REQUIRED' if not any(delta.values())
                else 'COUNT_MISMATCH_REVIEW_REQUIRED',
                message='未發現品項數量差異，仍需人工確認。' if not any(delta.values())
                else '偵測數量與預期不符，請人工確認。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'mps'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists. Choose a new run name; do not overwrite.')
    if sha(args.model) != MODEL_SHA256:
        parser.error('Model hash does not match the frozen baseline')
    config = json.loads(args.config.read_text())
    expected = config['expected_counts']
    validate_counts(expected)
    cases = config['cases']
    if not cases or len({c['image'] for c in cases}) != len(cases):
        parser.error('Cases must be nonempty and unique')
    for case in cases:
        if Path(case['image']).name != case['image']:
            parser.error('Image names must not contain directories')
        validate_counts(case['reference_counts'])
    originals = {c['image']: sha(args.images / c['image']) for c in cases}
    config_hash = sha(args.config)
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    if model.names != dict(enumerate(CLASSES)):
        parser.error('Unexpected model class mapping')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'annotated').mkdir()
    settings = dict(imgsz=640, conf=.25, iou=.7, max_det=300, rect=True,
                    device=args.device, half=False, augment=False, verbose=False, save=False)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  model_sha256=MODEL_SHA256, config_sha256=config_hash,
                  script_sha256=sha(Path(__file__)), input_sha256=originals,
                  python=sys.version, platform=platform.platform(),
                  packages={p: version(p) for p in ('torch', 'torchvision', 'ultralytics', 'Pillow')},
                  expected_counts=expected, settings=settings, images=[],
                  note='Counts-only staged demo; reference counts never enter inference. No bbox GT or mAP. '
                       'No automatic image-quality gate, unknown-drug detection, or clinical release. '
                       'A zero prediction count is not proof of physical absence. Errors are not count results.')
    errors = 0
    for case in cases:
        name = case['image']
        record = dict(image=name, reference_counts=case['reference_counts'])
        try:
            with Image.open(args.images / name) as raw:
                record['raw_size'] = list(raw.size)
                record['exif_orientation'] = raw.getexif().get(274, 1)
                image = ImageOps.exif_transpose(raw).convert('RGB')
                record['oriented_size'] = list(image.size)
                image.load()
            result = model.predict(source=image, **settings)[0]
            detected = dict.fromkeys(CLASSES, 0)
            predictions = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                detected[CLASSES[class_id]] += 1
                predictions.append(dict(class_name=CLASSES[class_id], confidence=float(box.conf.item()),
                                        xyxy=box.xyxy[0].tolist()))
            record.update(detected_counts=detected, predictions=predictions,
                          comparison=compare_counts(expected, detected),
                          counts_match_reference=detected == case['reference_counts'],
                          stage_ms=result.speed)
            result.save(filename=str(args.output / 'annotated' / (Path(name).stem + '.png')))
            print(name, detected, record['comparison']['message'])
        except Exception as exc:
            errors += 1
            record.update(status='PROCESSING_ERROR_REVIEW_REQUIRED', error=str(exc),
                          message='影像處理或推論失敗，無法比對，請人工確認。')
            print(name, record['message'], str(exc), file=sys.stderr)
        report['images'].append(record)
    if sha(args.model) != MODEL_SHA256 or sha(args.config) != config_hash or any(
            sha(args.images / name) != digest for name, digest in originals.items()):
        raise RuntimeError('Input integrity changed during this run; do not use results')
    report['summary'] = dict(images=len(cases), processing_errors=errors,
                            exact_count_images=sum(r.get('counts_match_reference', False) for r in report['images']))
    (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 預期品項與數量比對示範', '',
             '本次為五張安排案例的數量流程測試，不是定位或臨床安全性評估。',
             '影像品質須由人工確認；此版沒有自動模糊拒絕或未知藥物辨識。', '',
             '共同預期數量：四類各 1 顆。' if all(v == 1 for v in expected.values()) else str(expected), '',
             '|照片|品項|預期|偵測|差異|', '|---|---|---:|---:|---:|']
    for r in report['images']:
        if 'comparison' not in r:
            lines.append(f"|{r['image']}|處理失敗|—|—|不得視為正常結果|")
            continue
        for c in CLASSES:
            lines.append(f"|{r['image']}|{c}|{expected[c]}|{r['detected_counts'][c]}|{r['comparison']['delta'][c]:+d}|")
    lines += ['', '## 個案結論', '']
    for r in report['images']:
        msg = r['comparison']['message'] if 'comparison' in r else r['message']
        lines.append(f"- {r['image']}：{msg}")
    lines += ['', '預測框數不一定等於實際藥品數量；有疑義一律人工確認。原始照片未覆寫。',
              '速度為本次推論階段紀錄，非拍照到輸出的完整延遲。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print('Saved:', args.output)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
