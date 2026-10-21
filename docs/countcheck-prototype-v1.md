# 預期品項與數量比對小原型

2026-10-21。五張手機照片的安排案例，不作臨床放行依據。

## 已完成

- `docs/countcheck01.json`：共同模擬預期清單為 drug001 至 drug004 各一顆；逐張 reference_counts 來自使用者確認的實際擺放，僅供事後核對，未送入模型。
- `scripts/check_expected_counts.py`：固定原 best.pt SHA-256、conf=0.25、NMS IoU=0.7、imgsz=640、max_det=300。讀取時 EXIF transpose，不覆寫原圖。另存框圖、JSON、中文比對表及輸入與模型雜湊。
- `scripts/test_expected_counts.py`：以五種人工數量驗證差值規則，並拒絕缺漏類別、負數、布林及非整數數量。
- Mac CPU 首次執行輸出：`outputs/countcheck01-mac-cpu-conf025-v001/`。五張的預測數量皆與使用者提供的實際擺放一致；非五張皆與共同預期清單相同。全部框圖已人工檢視。
- Raspberry Pi 5 CPU 已於 2026-10-21 完成相同五張照片的執行，輸出已下載至 `outputs/countcheck01-pi-cpu-conf025-v001/`。處理錯誤 0，五張各類預測數量皆符合使用者提供的實際擺放，且與 Mac 一致。模型、五張原始影像、設定檔與執行程式的 SHA-256 兩端皆一致；這是數量層級核對，不宣稱框座標或未捨入張量完全相同。

## 本輪 Pi 證據

- 原型程式版本：Git `5d4ffaa`，沿用原始固定權重，不重新訓練、不調門檻。
- `result.json` SHA-256：`2150df1d332d1dd41fd5330973e82aeaaf98197312d70248a6776ed7609069f0`。
- 執行環境：Linux 6.18.50+rpt-rpi-2712 aarch64、torch 2.14.1+cpu、torchvision 0.29.1+cpu、Ultralytics 8.4.177、Pillow 12.3.0。系統核心與前次 40 張測試不同，歷史紀錄不覆蓋。
- `half=False` 為本輪保留的 FP32 舊參數；套件會轉為 `quantize=None` 並顯示停用預告。五張皆正常完成，未因警告改模型、精度或門檻。本次不修改程式，以保留實測版本。

|案例|預測數量 001 002 003 004|相對共同預期的差異|
|---|---|---|
|match|1 1 1 1|無差異 仍需人工確認|
|missing|1 0 1 1|002 少一顆|
|extra|2 1 1 1|001 多一顆|
|substitute|1 0 1 2|002 少一顆 004 多一顆|
|empty|0 0 0 0|四類皆未偵測到|

## 尚未完成與限制

本輪使用手機靜態照片，不是 Pi 相機或動態出藥驗證。無人工框 GT，因此不計算定位正確率或 mAP。這是五個開發示範，不宣稱獨立泛化效能或臨床準確率。沒有自動影像品質拒絕、未知藥物辨識、處方系統串接或動態事件計數。零預測不保證實際沒有藥品，數量一致也不證明藥包正確。若影像處理／推論失敗，明確輸出 PROCESSING_ERROR_REVIEW_REQUIRED，並以非零退出碼結束，不當成正常零顆結果。輸出路徑已存在即拒絕覆寫。

## 在 Mac 本機終端機傳送

本輪已透過 `sherry.local` 連線成功，避免使用已變動的舊 IP。若名稱無法解析，需先在 Pi 以 hostname -I 確認當下 IP。首次 SSH 連線須核對主機；密碼只在自己的終端機輸入。下列為重現步驟，已完成者不用重跑；輸出若已存在，應另取新 run 名稱。

```bash
ssh sherry@sherry.local 'mkdir -p /home/sherry/pill-review-pi/countcheck01'
scp /Users/sherry/projects/pill-review-assistant/data/raw/countcheck01/*.jpg sherry@sherry.local:/home/sherry/pill-review-pi/countcheck01/
scp /Users/sherry/projects/pill-review-assistant/scripts/check_expected_counts.py /Users/sherry/projects/pill-review-assistant/docs/countcheck01.json sherry@sherry.local:/home/sherry/pill-review-pi/
ssh sherry@sherry.local
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
scp -r sherry@sherry.local:/home/sherry/pill-review-pi/outputs/countcheck01-pi-cpu-conf025-v001 /Users/sherry/projects/pill-review-assistant/outputs/
```

原始照片、模型與 outputs 被 Git 忽略，仍需另外私人備份。Git 僅保存程式、案例設定及文字紀錄。本紀錄已依下載的 Pi 完整結果更新。
