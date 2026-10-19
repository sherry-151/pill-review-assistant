"""Export and evaluate exploratory controls, without training or modifying originals."""
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
import torch
import ultralytics

from audit_challenge import iou, read

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/control-yolo-v001'
OUT = ROOT / 'outputs/control-eval-conf025-v001'
MODEL = ROOT / 'outputs/baseline-20261016-150812-917344/weights/best.pt'
CLASSES = ['drug001', 'drug002', 'drug003', 'drug004']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def matching(gt, pred):
    edges = {i: sorted([j for j,p in enumerate(pred)
                       if int(g[0]) == int(p[0]) and iou(g,p) >= .5],
                       key=lambda j: iou(g,pred[j]), reverse=True)
             for i,g in enumerate(gt)}
    match = {}
    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match or assign(match[j], seen):
                match[j] = i
                return True
        return False
    for i in range(len(gt)):
        assign(i, set())
    return match

def preview(photo, gt, pred, dest):
    with Image.open(photo) as im:
        im = im.convert('RGB')
        ow,oh=im.size
        boxes=gt+pred
        left=max(0,int(min((b[1]-b[3]/2)*ow for b in boxes))-180)
        top=max(0,int(min((b[2]-b[4]/2)*oh for b in boxes))-180)
        right=min(ow,int(max((b[1]+b[3]/2)*ow for b in boxes))+180)
        bottom=min(oh,int(max((b[2]+b[4]/2)*oh for b in boxes))+180)
        im=im.crop((left,top,right,bottom))
        def crop_boxes(items):
            return [[b[0],(b[1]*ow-left)/(right-left),(b[2]*oh-top)/(bottom-top),
                     b[3]*ow/(right-left),b[4]*oh/(bottom-top)]+b[5:] for b in items]
        gt,pred=crop_boxes(gt),crop_boxes(pred)
        im.thumbnail((650, 850))
        w,h = im.size
        canvas = Image.new('RGB', (w*2, h+35), 'white')
        for col, boxes in enumerate((gt,pred)):
            canvas.paste(im, (col*w,35))
            draw = ImageDraw.Draw(canvas)
            draw.text((col*w+8,8), 'GROUND TRUTH (detail crop)' if col==0 else 'PRED conf>=0.25 (detail crop)', fill='black')
            for k,b in enumerate(boxes):
                _,x,y,bw,bh = b[:5]
                xy = (col*w+(x-bw/2)*w,35+(y-bh/2)*h,
                      col*w+(x+bw/2)*w,35+(y+bh/2)*h)
                color = ['red','blue','orange','purple'][int(b[0])]
                draw.rectangle(xy, outline=color, width=3)
                text = CLASSES[int(b[0])] + (f' {b[5]:.3f}' if len(b)>5 else '')
                draw.text((col*w+8,40+k*18),text,fill=color,stroke_width=1,stroke_fill='white')
        canvas.save(dest)

def main():
    if DEST.exists() or OUT.exists():
        raise SystemExit('Version exists; refuse to overwrite. Preserve existing results.')
    records = []
    source_hashes = {}
    for folder, count, expected in [('clean_touch01',10,Counter({'drug001':1,'drug003':1})),
                                    ('occlusion_control01',15,Counter({'drug002':1,'drug004':1}))]:
        source = ROOT/'data/raw'/folder
        photos = sorted(source.glob('*.jpg'))
        assert len(photos)==count and len(list(source.glob('*.json')))==count
        for photo in photos:
            annotation = photo.with_suffix('.json')
            data = json.loads(annotation.read_text())
            with Image.open(photo) as im:
                orientation = im.getexif().get(274,1)
                w,h = ImageOps.exif_transpose(im).size
            assert (w,h)==(data['imageWidth'],data['imageHeight']), photo
            assert Counter(s['label'] for s in data['shapes'])==expected, photo
            lines=[]
            for s in data['shapes']:
                assert s['shape_type']=='rectangle' and len(s['points'])==2
                (x1,y1),(x2,y2)=s['points']
                x1,x2=sorted((x1,x2)); y1,y2=sorted((y1,y2))
                assert 0<=x1<x2<=w and 0<=y1<y2<=h
                box=[(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h]
                lines.append(str(CLASSES.index(s['label']))+' '+' '.join(f'{v:.8f}' for v in box))
            for p in (photo,annotation):
                source_hashes[str(p.relative_to(ROOT))]=digest(p)
            parts=photo.stem.split('_')
            group=('clean_'+parts[1]) if folder=='clean_touch01' else parts[2]
            records.append(dict(source=str(photo.relative_to(ROOT)),group=group,
                                pair=parts[1] if folder=='occlusion_control01' else None,
                                original_exif_orientation=orientation, width=w,height=h,lines=lines))
    images=DEST/'images/test'; labels=DEST/'labels/test'
    images.mkdir(parents=True); labels.mkdir(parents=True)
    for r in records:
        photo=ROOT/r['source']
        with Image.open(photo) as im:
            ImageOps.exif_transpose(im).convert('RGB').save(images/photo.name,quality=95)
        with Image.open(images/photo.name) as im:
            assert im.size==(r['width'],r['height']) and im.getexif().get(274,1)==1
        (labels/(photo.stem+'.txt')).write_text('\n'.join(r['lines'])+'\n')
        r['export_sha256']=digest(images/photo.name)
    # Required schema fields point at this evaluation set; never use it for training.
    config=DEST/'evaluation-only.yaml'
    config.write_text(f'path: {json.dumps(str(DEST))}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(CLASSES)))
    manifest=dict(purpose='Exploratory evaluation only; not training or blind testing',
                  annotation_policy='Visible-part bounding boxes; no inferred hidden extent',
                  source_sha256=source_hashes,images=records)
    (ROOT/'data/splits/control-v001-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    model_hash=digest(MODEL)
    model=YOLO(str(MODEL))
    model.val(data=str(config),split='test',imgsz=640,batch=8,device='mps',workers=0,
              conf=.25,iou=.7,max_det=300,plots=True,save_txt=True,save_conf=True,
              project=str(OUT.parent),name=OUT.name,exist_ok=False)
    assert OUT.exists()
    assert digest(MODEL)==model_hash
    assert all(digest(ROOT/p)==h for p,h in source_hashes.items()), 'Original modified'
    audit=OUT/'audit'; audit.mkdir()
    previews=OUT/'per-image'; previews.mkdir()
    rows=[]; groups=defaultdict(Counter)
    for r in records:
        stem=Path(r['source']).stem
        gt=read(labels/(stem+'.txt')); pred=read(OUT/'labels'/(stem+'.txt'))
        match=matching(gt,pred)
        missed=[i for i in range(len(gt)) if i not in match.values()]
        extra=[j for j in range(len(pred)) if j not in match]
        row=dict(image=stem+'.jpg',group=r['group'],pair=r['pair'],gt=len(gt),pred=len(pred),
                 matched=len(match),unmatched_gt=len(missed),unmatched_pred=len(extra),
                 exact=not missed and not extra,extra_predictions=[pred[j] for j in extra])
        if r['pair']:
            target=next(g for g in gt if int(g[0])==1)
            candidates=[p for p in pred if int(p[0])==1]
            best=max(candidates,key=lambda p:iou(target,p),default=None)
            row['drug002']=dict(predicted=bool(candidates),
                best_iou=iou(target,best) if best else None,
                confidence=best[5] if best else None,
                localized=any(iou(target,p)>=.5 for p in candidates))
        rows.append(row)
        groups[r['group']].update(images=1,exact=int(row['exact']),missed=len(missed),extra=len(extra))
        preview(images/(stem+'.jpg'),gt,pred,previews/(stem+'.png'))
    result=dict(model_sha256=model_hash,confidence=.25,matching_iou=.5,
                matching='same-class maximum-cardinality one-to-one',groups=dict(groups),images=rows,
                note='Unmatched boxes may reflect localization/classification errors or extras; inspect previews.',
                python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,
                os=platform.platform(),device='mps',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    (audit/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    report=['# 固定模型探索性評估', '', 'conf=0.25；同類別一對一配對 IoU≥0.50；未重訓，原始照片與標註、模型雜湊確認未變。',
            '', '左側人工標註，右側固定門檻預測：每張對照圖在 `../per-image/`。', '',
            '|組別|照片|整張框與類別全對|未配對人工框|未配對預測框|','|---|---:|---:|---:|---:|']
    for group,c in groups.items():
        report.append(f"|{group}|{c['images']}|{c['exact']}|{c['missed']}|{c['extra']}|")
    report+=['','## 遮蔽對照 drug002','', '|組別|狀態|有同類預測|最佳可見框 IoU|該框信心值|IoU≥0.50|','|---|---|---|---:|---:|---|']
    for row in rows:
        if not row['pair']: continue
        d=row['drug002']; fmt=lambda x:'—' if x is None else f'{x:.3f}'
        report.append(f"|{row['pair']}|{row['group']}|{d['predicted']}|{fmt(d['best_iou'])}|{fmt(d['confidence'])}|{d['localized']}|")
    report+=['','## 需查看的照片','']
    report += [f"- [{r['image']}](../per-image/{Path(r['image']).stem}.png)：未配對人工框 {r['unmatched_gt']}，未配對預測框 {r['unmatched_pred']}。" for r in rows if not r['exact']]
    report+=['','少量樣本、同場景配對不是獨立大型測試；遮蔽比例未量測，姿態也可能改變。乾淨接觸不是嚴格有毛／無毛配對，不能推論毛髮因果。信心值不是正確機率。']
    (audit/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(groups=dict(groups),errors=[r for r in rows if not r['exact']]),indent=2))
    print('Report:',audit/'report.md')

if __name__=='__main__':
    main()
