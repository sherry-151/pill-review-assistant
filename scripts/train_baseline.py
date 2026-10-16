"""Train a fixed baseline using train/val only. Run from the project venv."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='One epoch to check the workflow; not a final result')
    parser.add_argument('--device', default='mps', choices=['mps','cpu'])
    args = parser.parse_args()
    dataset = ROOT/'data/yolo-v001/dataset.yaml'
    manifest = ROOT/'data/splits/yolo-v001-manifest.json'
    if not dataset.exists() or not manifest.exists():
        raise SystemExit('Run prepare_yolo.py first.')
    dirty = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if dirty.strip():
        raise SystemExit('Commit project changes before training so the exact code is recorded.')
    import torch
    from ultralytics import YOLO
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS unavailable. Rerun with --device cpu.')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    name = ('smoke-' if args.smoke else 'baseline-')+stamp
    folder = ROOT/'outputs'/name
    folder.mkdir(parents=True, exist_ok=False)
    env = subprocess.check_output([sys.executable,str(ROOT/'scripts/record_environment.py')],cwd=ROOT,text=True)
    report = json.loads(env); report['actual_training_device']=args.device
    (folder/'environment.json').write_text(json.dumps(report,indent=2)+'\n')
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (folder/'requirements.lock.txt').write_text(freeze)
    config = dict(data=str(dataset),epochs=1 if args.smoke else 50,imgsz=640,batch=8,device=args.device,workers=0,seed=42,deterministic=True,optimizer='AdamW',lr0=0.001,patience=15,mosaic=0.0,mixup=0.0,hsv_h=0.0,hsv_s=0.0,hsv_v=0.0,degrees=0.0,translate=0.0,scale=0.0,fliplr=0.0,flipud=0.0,amp=False,project=str(ROOT/'outputs'),name=name,exist_ok=True,plots=True)
    record = dict(status='starting',smoke_only=args.smoke,git_commit=report['git_commit'],manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),model='yolov8n.pt',arguments=config,note='Simple transfer-learning baseline with augmentations disabled; no test-set evaluation.')
    record_path=folder/'experiment.json'
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    try:
        model=YOLO('yolov8n.pt')  # Downloads official pretrained weights on first use.
        initial=Path(model.ckpt_path)
        record['initial_weights_sha256']=hashlib.sha256(initial.read_bytes()).hexdigest()
        record_path.write_text(json.dumps(record,indent=2)+'\n')
        model.train(**config)
        record['status']='completed'
        best=folder/'weights/best.pt'
        if best.exists(): record['best_weights_sha256']=hashlib.sha256(best.read_bytes()).hexdigest()
    except Exception as error:
        record['status']='failed'; record['error']=str(error)
        raise
    finally:
        record_path.write_text(json.dumps(record,indent=2)+'\n')
    print('Results:',folder)

if __name__ == '__main__':
    main()
