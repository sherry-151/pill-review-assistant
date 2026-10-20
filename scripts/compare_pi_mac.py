"""Compare saved Pi test results with a matched Mac MPS evaluation, no training."""
import json
import platform
from importlib.metadata import version
from pathlib import Path
from ultralytics import YOLO
from evaluate_pi_test import sha, read, iou

ROOT=Path(__file__).resolve().parents[1]
PI=ROOT/'outputs/test40-cpu-conf025-v001/audit/audit.json'
OUT=ROOT/'outputs/mac-test40-conf025-v001'
MODEL=ROOT/'outputs/baseline-20261016-150812-917344/weights/best.pt'

def main():
    pi=json.loads(PI.read_text())
    assert sha(MODEL)==pi['model_sha256']
    for name,h in pi['input_sha256'].items():
        assert sha(ROOT/'data/yolo-v001'/name.removeprefix('test-v001/'))==h, name
    assert sha(ROOT/'scripts/evaluate_pi_test.py')==pi['script_sha256']
    if OUT.exists():
        raise SystemExit('Mac comparison output exists; preserve previous results.')
    args={k:v for k,v in pi['arguments'].items() if k not in ('data','device','project','name')}
    args.update(data=str(ROOT/'data/yolo-v001/dataset-with-test.yaml'),device='mps',project=str(OUT.parent),name=OUT.name)
    result=YOLO(str(MODEL)).val(**args)
    rows=[]
    for r in pi['images']:
        gt=read(ROOT/'data/yolo-v001/labels/test'/Path(r['image']).with_suffix('.txt'))
        mac=read(OUT/'labels'/Path(r['image']).with_suffix('.txt'))
        p=r['predictions']
        matched=any(g[0]==q[0] and iou(g,q)>=.5 for g in gt for q in mac)
        row=dict(image=r['image'],pi_exact=r['exact'],mac_exact=matched and len(mac)==1,
                 class_count_equal=sorted(q[0] for q in p)==sorted(q[0] for q in mac))
        if len(p)==len(mac)==1:
            row.update(pi_mac_box_iou=iou(p[0],mac[0]),confidence_abs_difference=abs(p[0][5]-mac[0][5]),
                       max_normalized_coordinate_abs_difference=max(abs(a-b) for a,b in zip(p[0][1:5],mac[0][1:5])))
        rows.append(row)
    summary=dict(images=len(rows),pi_exact=sum(r['pi_exact'] for r in rows),mac_exact=sum(r['mac_exact'] for r in rows),
                 same_class_count_images=sum(r['class_count_equal'] for r in rows),
                 min_pi_mac_box_iou=min(r['pi_mac_box_iou'] for r in rows if 'pi_mac_box_iou' in r),
                 max_confidence_abs_difference=max(r['confidence_abs_difference'] for r in rows if 'confidence_abs_difference' in r),
                 max_normalized_coordinate_abs_difference=max(r['max_normalized_coordinate_abs_difference'] for r in rows if 'max_normalized_coordinate_abs_difference' in r))
    report=dict(summary=summary,images=rows,arguments=args,metrics=result.results_dict,speed_ms=result.speed,
                os=platform.platform(),packages={n:version(n) for n in ('torch','torchvision','ultralytics','numpy','opencv-python')},
                model_sha256=sha(MODEL),pi_report_sha256=sha(PI),comparison_script_sha256=sha(Path(__file__)),
                note='Same inputs and thresholds, different device, Python and dependency versions. Not bitwise or general clinical equivalence; one evaluation run, not end-to-end timing.')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPARISON SUMMARY:',json.dumps(summary,indent=2))

if __name__=='__main__':main()
