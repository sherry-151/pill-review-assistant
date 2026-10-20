"""Fixed 40-image CPU evaluation; writes a new run, never trains or overwrites."""
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path.home() / 'pill-review-pi'
OUT = ROOT / 'outputs/test40-cpu-conf025-v001'
MODEL = ROOT / 'models/best.pt'
EXPECTED_MODEL = 'd195469586c0511b28c5052735372e609d5003c5923c3fb23ec207d22055a0f4'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def iou(a,b):
    x,y,w,h=a[1:5]; X,Y,W,H=b[1:5]
    overlap=max(0,min(x+w/2,X+W/2)-max(x-w/2,X-W/2))*max(0,min(y+h/2,Y+H/2)-max(y-h/2,Y-H/2))
    return overlap/(w*h+W*H-overlap)

def main():
    if OUT.exists():
        raise SystemExit('Run already exists. Preserve it; do not overwrite or delete it.')
    images=ROOT/'test-v001/images/test'; labels=ROOT/'test-v001/labels/test'
    photos=sorted(images.glob('*.jpg'))
    assert len(photos)==40 and len(list(labels.glob('*.txt')))==40, 'Expected 40 images and 40 GT labels'
    assert MODEL.exists() and sha(MODEL)==EXPECTED_MODEL, 'Model missing or hash mismatch'
    gt_counts=Counter(); originals={}
    for photo in photos:
        gt=labels/(photo.stem+'.txt')
        assert gt.exists(), photo
        rows=read(gt)
        assert len(rows)==1 and len(rows[0])==5 and rows[0][0] in range(4), photo
        gt_counts[int(rows[0][0])]+=1
        for path in (photo,gt): originals[str(path.relative_to(ROOT))]=sha(path)
    assert gt_counts==Counter({0:10,1:10,2:10,3:10}), gt_counts
    from ultralytics import YOLO
    config=dict(data=str(ROOT/'test-v001/evaluation.yaml'),split='test',imgsz=640,batch=8,
                device='cpu',workers=0,conf=.25,iou=.7,max_det=300,rect=True,
                plots=True,save_txt=True,save_conf=True,project=str(OUT.parent),name=OUT.name,exist_ok=False)
    result=YOLO(str(MODEL)).val(**config)
    audit=OUT/'audit'; audit.mkdir()
    records=[]
    for photo in photos:
        gt=read(labels/(photo.stem+'.txt')); pred=read(OUT/'labels'/(photo.stem+'.txt'))
        # This fixed dataset has exactly one GT object per image.
        candidates=[j for j,p in enumerate(pred) if p[0]==gt[0][0] and iou(gt[0],p)>=.5]
        match=max(candidates,key=lambda j:iou(gt[0],pred[j]),default=None)
        records.append(dict(image=photo.name,gt_class=int(gt[0][0]),predictions=pred,
                            matched=int(match is not None),unmatched_gt=int(match is None),
                            unmatched_pred=len(pred)-int(match is not None),
                            exact=match is not None and len(pred)==1,
                            matched_iou=iou(gt[0],pred[match]) if match is not None else None))
    assert sha(MODEL)==EXPECTED_MODEL
    assert all(sha(ROOT/p)==h for p,h in originals.items()), 'Inputs changed'
    summary=dict(images=40,gt=40,pred=sum(len(r['predictions']) for r in records),
                 exact_images=sum(r['exact'] for r in records),
                 unmatched_gt=sum(r['unmatched_gt'] for r in records),
                 unmatched_pred=sum(r['unmatched_pred'] for r in records))
    report=dict(summary=summary,images=records,arguments=config,matching_iou=.5,
                pairing='same-class one-to-one; this dataset has one GT per image',
                created_utc=datetime.now(timezone.utc).isoformat(),model_sha256=EXPECTED_MODEL,
                input_sha256=originals,script_sha256=sha(Path(__file__)),python=sys.version,
                os=platform.platform(),architecture=platform.machine(),
                packages={name:version(name) for name in ('torch','torchvision','ultralytics','numpy','opencv-python')},
                metrics=result.results_dict,speed_ms=result.speed,
                timing_note='Ultralytics stage averages for this evaluation run; excludes capture/network and full startup. Not a steady-state or end-to-end benchmark.')
    (audit/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (audit/'requirements-pi.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print('\nFIXED CONFIDENCE COUNTS:',json.dumps(summary))
    print('Audit saved:',audit/'audit.json')

if __name__=='__main__':
    main()
