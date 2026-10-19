"""Audit saved predictions at the fixed 0.25 confidence operating point."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRED=ROOT/'outputs/operating-challenge-conf025/labels'
GT=ROOT/'data/challenge-yolo-v001/labels/test'
OUT=ROOT/'outputs/operating-challenge-conf025/audit'

def read(path):
    return [list(map(float,line.split())) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def iou(a,b):
    x,y,w,h=a[1:5]; X,Y,W,H=b[1:5]
    area=max(0,min(x+w/2,X+W/2)-max(x-w/2,X-W/2))*max(0,min(y+h/2,Y+H/2)-max(y-h/2,Y-H/2))
    return area/(w*h+W*H-area)

def main():
    assert GT.exists() and PRED.exists(), 'Saved labels missing'
    rows=[]; groups=defaultdict(Counter)
    for f in sorted(GT.glob('*.txt')):
        a=read(f); b=read(PRED/f.name)
        # Maximum-cardinality one-to-one matching, restricted to correct class and IoU>=0.50.
        edges={i:sorted([j for j,p in enumerate(b) if int(g[0])==int(p[0]) and iou(g,p)>=0.5],key=lambda j:iou(g,b[j]),reverse=True) for i,g in enumerate(a)}
        match={}
        def assign(i,seen):
            for j in edges[i]:
                if j in seen: continue
                seen.add(j)
                if j not in match or assign(match[j],seen):
                    match[j]=i; return True
            return False
        for i in range(len(a)): assign(i,set())
        missed=[i for i in range(len(a)) if i not in match.values()]
        extra=[j for j in range(len(b)) if j not in match]
        count_ok=Counter(int(x[0]) for x in a)==Counter(int(x[0]) for x in b)
        localization_ok=not missed and not extra
        group=f.stem.split('_')[1]
        row=dict(image=f.stem+'.jpg',group=group,gt=len(a),pred=len(b),matched=len(match),unmatched_gt=len(missed),unmatched_pred=len(extra),class_count_exact=count_ok,all_class_iou50_exact=localization_ok,unmatched_gt_classes=[f'drug{int(a[i][0])+1:03d}' for i in missed],unmatched_predictions=[dict(label=f'drug{int(b[j][0])+1:03d}',confidence=b[j][5] if len(b[j])>5 else None) for j in extra])
        rows.append(row)
        groups[group].update(dict(images=1,gt=len(a),pred=len(b),matched=len(match),unmatched_gt=len(missed),unmatched_pred=len(extra),class_count_exact=int(count_ok),all_class_iou50_exact=int(localization_ok)))
    assert len(rows)==40
    result=dict(confidence=0.25,matching_iou=0.5,matching='maximum-cardinality, same-class, one-to-one',note='Unmatched ground truth/predictions may reflect misclassification or poor localization, not only missed/extra objects.',groups=dict(groups),errors=[r for r in rows if not r['all_class_iou50_exact']],images=rows)
    OUT.mkdir(exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'per-image.csv').open('w',newline='') as stream:
        fields=list(rows[0]); writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
