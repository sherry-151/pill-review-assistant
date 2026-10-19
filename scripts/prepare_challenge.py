"""Export challenge01 without changing the baseline train/val/test dataset."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['drug001', 'drug002', 'drug003', 'drug004']
DEST = ROOT / 'data/challenge-yolo-v001'

def main():
    if DEST.exists():
        raise SystemExit('Output exists. Preserve this dataset version; do not overwrite it.')
    records=[]
    photos=sorted((ROOT/'data/raw/challenge01').glob('*.jpg'))
    assert len(photos)==40, 'Expected 40 challenge photos'
    for photo in photos:
        annotation=photo.with_suffix('.json')
        data=json.loads(annotation.read_text())
        with Image.open(photo) as im:
            size=ImageOps.exif_transpose(im).size
        assert size==(data['imageWidth'],data['imageHeight']), f'{photo}: orientation mismatch'
        lines=[]
        for shape in data['shapes']:
            assert shape['shape_type']=='rectangle' and shape['label'] in CLASSES
            assert len(shape['points'])==2
            (x1,y1),(x2,y2)=shape['points']; x1,x2=sorted([x1,x2]); y1,y2=sorted([y1,y2])
            w,h=size
            assert 0<=x1<x2<=w and 0<=y1<y2<=h
            box=[(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h]
            lines.append(str(CLASSES.index(shape['label']))+' '+' '.join(f'{v:.8f}' for v in box))
        assert lines, f'{photo}: no annotations'
        records.append(dict(source=str(photo.relative_to(ROOT)),group=photo.stem.split('_')[1],image_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),lines=lines))
    assert Counter(r['group'] for r in records)==dict(side=10,separate=15,touch=10,overlap=5)
    images=DEST/'images/test'; labels=DEST/'labels/test'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    for r in records:
        photo=ROOT/r['source']
        with Image.open(photo) as im:
            ImageOps.exif_transpose(im).convert('RGB').save(images/photo.name,quality=95)
        (labels/(photo.stem+'.txt')).write_text('\n'.join(r['lines'])+'\n')
    # No training images are exported; only split=test should be used with this config.
    config=f'path: {json.dumps(str(DEST))}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n'
    config+=''.join(f'  {i}: {name}\n' for i,name in enumerate(CLASSES))
    (DEST/'challenge.yaml').write_text(config)
    (ROOT/'data/splits/challenge-v001-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print('Challenge export complete: 40 images,',sum(len(r['lines']) for r in records),'boxes')
    print('Evaluation only; do not train using challenge.yaml.')

if __name__=='__main__':
    main()
