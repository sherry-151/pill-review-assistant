"""Describe approximate pill scale; no inference, cropping, training or label edits."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measures(box, size):
    width, height = size
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Box outside image')
    bw, bh = x2-x1, y2-y1
    return dict(long_edge_fraction=max(bw, bh)/max(width, height),
                bbox_area_fraction=bw*bh/(width*height),
                approximate_long_edge_pixels_at_imgsz640=max(bw, bh)*640/max(width, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous results; choose a new output file')
    manifest = args.root/'data/splits/yolo-v001-manifest.json'
    estimates = args.root/'data/splits/camera-v3-oatmeal01-size-estimates.json'
    phone = json.loads(manifest.read_text())
    camera = json.loads(estimates.read_text())
    from PIL import Image
    rows = []
    inputs = {}
    for sample in camera['images']:
        image = args.root/camera['source_dir']/sample['image']
        with Image.open(image) as im:
            if list(im.size) != sample['size'] or im.getexif().get(274, 1) != 1:
                raise ValueError('Unexpected camera size/orientation')
        inputs[sample['image']] = sha(image)
        reference = [r for r in phone if r['split'] == 'train' and r['label'] == sample['class_name']]
        if len(reference) != 20:
            raise ValueError('Expected 20 baseline training annotations per class')
        values = [measures(r['bbox_xyxy'], [r['width'], r['height']]) for r in reference]
        medians = {k: statistics.median(v[k] for v in values) for k in values[0]}
        new = measures(sample['approx_bbox_xyxy'], sample['size'])
        rows.append(dict(class_name=sample['class_name'], phone_training_boxes=len(reference),
                         phone_median=medians, camera_approximate=new,
                         approximate_linear_scale_ratio=medians['long_edge_fraction']/new['long_edge_fraction']))
    report = dict(phone_manifest_sha256=sha(manifest), estimates_sha256=sha(estimates),
                  script_sha256=sha(Path(__file__)), camera_input_sha256=inputs,
                  method=camera['method'], rows=rows,
                  limitations='Phone: existing manual AABB annotations, training split only. '
                  'Camera: approximate visual AABBs, not detector outputs. '
                  'Normalized long edge is max bbox dimension / max image dimension. '
                  '640-pixel estimates omit integer rounding and padding; no model rerun. '
                  'AABB rotation and 3:4 vs 16:9 aspect ratio affect comparisons. '
                  'Not physical length, segmentation area, exact focal magnification, or proof of error causation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
