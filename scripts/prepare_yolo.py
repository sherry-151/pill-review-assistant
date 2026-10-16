"""Validate Labelme rectangles and export EXIF-normalized YOLO images."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/yolo-v001'
CLASSES = ['drug001', 'drug002', 'drug003', 'drug004']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if DEST.exists():
        raise SystemExit('Output already exists; preserve it and use a new dataset version for changes.')
    records, seen = [], {}
    for session, split, expected in [('01', 'train', 20), ('02', 'val', 10), ('03', 'test', 10)]:
        for label in CLASSES:
            folder = ROOT / f'data/raw/session{session}_{label}'
            photos = sorted(p for p in folder.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg'})
            assert len(photos) == expected, f'{folder}: expected {expected} photos'
            assert len(list(folder.glob('*.json'))) == expected, f'{folder}: JSON count'
            for photo in photos:
                annotation = photo.with_suffix('.json')
                d = json.loads(annotation.read_text())
                with Image.open(photo) as im:
                    orientation = im.getexif().get(274, 1)
                    corrected = ImageOps.exif_transpose(im).convert('RGB')
                    width, height = corrected.size
                    pixel_hash = hashlib.sha256(corrected.tobytes()).hexdigest()
                assert (width, height) == (d['imageWidth'], d['imageHeight']), f'{photo}: label/image orientation mismatch; do not guess coordinates'
                if pixel_hash in seen:
                    assert seen[pixel_hash] == split, f'{photo}: identical image pixels in different splits'
                seen[pixel_hash] = split
                assert len(d['shapes']) == 1, f'{photo}: expected one shape'
                shape = d['shapes'][0]
                assert shape['label'] == label and shape['shape_type'] == 'rectangle', f'{photo}: label/type'
                assert len(shape['points']) == 2, f'{photo}: rectangle points'
                (x1, y1), (x2, y2) = shape['points']
                x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f'{photo}: invalid box'
                values = [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]
                records.append(dict(source=str(photo.relative_to(ROOT)), annotation=str(annotation.relative_to(ROOT)), split=split, label=label, class_id=CLASSES.index(label), width=width, height=height, original_exif_orientation=orientation, source_sha256=sha(photo), annotation_sha256=sha(annotation), pixel_sha256=pixel_hash, bbox_xyxy=[x1,y1,x2,y2], yolo_xywh=values))
    # No output is written until all source files pass validation.
    for r in records:
        image_dir = DEST / 'images' / r['split']; label_dir = DEST / 'labels' / r['split']
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        source = ROOT / r['source']
        out = image_dir / (source.stem + '.jpg')
        with Image.open(source) as im:
            ImageOps.exif_transpose(im).convert('RGB').save(out, quality=95)
        text = str(r['class_id']) + ' ' + ' '.join(f'{v:.8f}' for v in r['yolo_xywh']) + '\n'
        target = label_dir / (source.stem + '.txt'); target.write_text(text)
        r['export_image_sha256'] = sha(out); r['export_label_sha256'] = sha(target)
    config = f'path: {json.dumps(str(DEST))}\ntrain: images/train\nval: images/val\n'
    config += 'names:\n' + ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASSES))
    (DEST/'dataset.yaml').write_text(config)
    (DEST/'dataset-with-test.yaml').write_text(config+'test: images/test\n')
    manifest = ROOT/'data/splits/yolo-v001-manifest.json'
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    # Only train/val samples are rendered; the held-out test is not used for model selection.
    sheet = Image.new('RGB', (1200, 840), 'white'); draw = ImageDraw.Draw(sheet)
    for row, split in enumerate(['train','val']):
        for col, label in enumerate(CLASSES):
            r = next(x for x in records if x['split']==split and x['label']==label)
            path = DEST/'images'/split/Path(r['source']).name
            with Image.open(path) as image:
                # Draw boxes reconstructed from the exported YOLO text, not the original coordinates.
                fields = (DEST/'labels'/split/(path.stem+'.txt')).read_text().split()
                xc,yc,bw,bh = map(float,fields[1:]); w,h=image.size
                ImageDraw.Draw(image).rectangle(((xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h),outline='red',width=8)
                image.thumbnail((290,390)); sheet.paste(image,(col*300,row*420+25))
            draw.text((col*300+5,row*420+5),split+' '+label,fill='black')
    sheet.save(DEST/'qa-train-val.jpg')
    print('Export complete:',dict(Counter(r['split'] for r in records)))
    print('EXIF orientations:',dict(Counter(r['original_exif_orientation'] for r in records)))
    print('Manifest:',manifest)
    print('Preview:',DEST/'qa-train-val.jpg')

if __name__ == '__main__':
    main()
