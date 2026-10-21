# 預期品項與數量比對小原型

2026-10-21。五張手機照片的安排案例，不作臨床放行依據。

## 已完成

- `docs/countcheck01.json`：共同模擬預期清單為 drug001 至 drug004 各一顆；逐張 reference_counts 來自使用者確認的實際擺放，僅供事後核對，未送入模型。
- `scripts/check_expected_counts.py`：固定原 best.pt SHA-256、conf=0.25、NMS IoU=0.7、imgsz=640、max_det=300。讀取時 EXIF transpose，不覆寫原圖。另存框圖、JSON、中文比對表及輸入與模型雜湊。
- `scripts/test_expected_counts.py`：以五種人工數量驗證差值規則，並拒絕缺漏類別、負數、布林及非整數數量。
- Mac CPU 首次執行輸出：`outputs/countcheck01-mac-cpu-conf025-v001/`。五張的預測數量皆與使用者提供的實際擺放一致；非五張皆與共同預期清單相同。全部框圖已人工檢視。

|案例|預測數量 001 002 003 004|相對共同預期的差異|
|---|---|---|
|match|1 1 1 1|無差異 仍需人工確認|
|missing|1 0 1 1|002 少一顆|
|extra|2 1 1 1|001 多一顆|
|substitute|1 0 1 2|002 少一顆 004 多一顆|
|empty|0 0 0 0|四類皆未偵測到|

## 尚未完成與限制

本輪尚未在 Pi 執行。無人工框 GT，因此不計算定位正確率或 mAP。這是五個開發示範，不宣稱獨立泛化效能或臨床準確率。沒有自動影像品質拒絕、未知藥物辨識、處方系統串接或動態事件計數。零預測不保證實際沒有藥品，數量一致也不證明藥包正確。若影像處理／推論失敗，明確輸出 PROCESSING_ERROR_REVIEW_REQUIRED，並以非零退出碼結束，不當成正常零顆結果。輸出路徑已存在即拒絕覆寫。

## 在 Mac 本機終端機傳送

IP 若已變更，請替換下列位址。首次 SSH 連線須核對主機；密碼只在自己的終端機輸入。

```bash
ssh sherry@192.168.68.69 'mkdir -p /home/sherry/pill-review-pi/countcheck01'
scp /Users/sherry/projects/pill-review-assistant/data/raw/countcheck01/*.jpg sherry@192.168.68.69:/home/sherry/pill-review-pi/countcheck01/
scp /Users/sherry/projects/pill-review-assistant/scripts/check_expected_counts.py /Users/sherry/projects/pill-review-assistant/docs/countcheck01.json sherry@192.168.68.69:/home/sherry/pill-review-pi/
ssh sherry@192.168.68.69
```

## 在樹莓派 SSH 終端機執行

提示字元應含 `sherry@sherry`。使用已存在的 Pi CPU 虛擬環境與原本權重。

```bash
source ~/pill-inference-venv/bin/activate
python ~/pill-review-pi/check_expected_counts.py \
  --model ~/pill-review-pi/models/best.pt \
  --images ~/pill-review-pi/countcheck01 \
  --config ~/pill-review-pi/countcheck01.json \
  --output ~/pill-review-pi/outputs/countcheck01-pi-cpu-conf025-v001 \
  --device cpu
```

完成後，在 Mac 的另一個本機終端機下載至新位置；不要把 Mac 版和 Pi 版輸出混在一起。

```bash
scp -r sherry@192.168.68.69:/home/sherry/pill-review-pi/outputs/countcheck01-pi-cpu-conf025-v001 /Users/sherry/projects/pill-review-assistant/outputs/
```

原始照片、模型與 outputs 被 Git 忽略，仍需另外私人備份。Git 僅保存程式、案例設定及文字紀錄；Pi 執行結果須於實測後另行更新。
