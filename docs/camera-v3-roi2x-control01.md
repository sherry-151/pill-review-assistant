# 固定中央取景對照結果

日期：2026-10-25。依預先訂定計畫只測試一次，未按藥品改裁切位置、調門檻或重新訓練。執行裝置為 Mac CPU。

## 輸入與完整性

五張燕麥背景原圖位於 `data/raw/camera-v3-燕麥背景/`，保留不變；來源 SHA-256 與先前整張推論紀錄全部一致。各張均以相同座標 `[1152, 648, 3456, 1944]` 取 4608 × 2592 原圖中央 1/2 寬、1/2 高，得到 2304 × 1296 衍生圖。空背景也套同一區域，沒有利用辨識框決定位置。

衍生圖另存 `data/derived/camera-v3-oatmeal01-roi2x/`，使用無損 PNG，不重採樣、不銳化、不調色。逐張核對存檔像素與原圖解碼後的該區域完全相同。五張衍生影像在推論前均已檢視，四顆藥物完整保留、未被切斷。`data/derived/` 已加入 Git 忽略，避免影像上傳。

準備程式為 `scripts/prepare_camera_roi.py`，來源及衍生圖雜湊、固定座標與程式雜湊保存於 `outputs/camera-v3-oatmeal01-roi2x-preparation-v001/manifest.json`。案例清單為 `data/splits/camera-v3-oatmeal01-roi2x.json`。人工類別僅用於事後核對，不輸入模型。

## 固定設定與逐張結果

原 best.pt（SHA-256 `d195469586c0511b28c5052735372e609d5003c5923c3fb23ec207d22055a0f4`）、conf 0.25、imgsz 640、NMS IoU 0.7、CPU FP32、max_det 300、rect True、augment False、agnostic_nms False，與原整張測試相同。新增裁切是不同的輸入前處理，不宣稱原整張照片通過。

|實際擺放|原整張燕麥背景|固定中央取景|本次判讀|
|---|---|---|---|
|drug001 一顆|drug002 一框，0.964817|drug001 一框，0.964251|由辨錯變為正確類別及數量|
|drug002 一顆|drug002 一框，0.258121|drug002 一框，0.694554|維持正確類別及數量|
|drug003 一顆|零框|drug003 一框，0.872151|由漏辨變為正確類別及數量|
|drug004 一顆|零框|零框|仍漏辨，膠囊沒有被裁掉|
|空背景|零框|零框|本張無背景誤報|

已逐張查看五張推論輸出，三個預測框皆位於相應藥物上，沒有多框。有藥四張中由 1／4 變為 3／4 符合；含空背景則由 2／5 變為 4／5 符合。處理錯誤零，五張 PNG 與五份預測標籤全部保存（零框仍有空標籤檔）。沒有人工定位真值，不報告定位 IoU、precision/recall 或 mAP；這些計數不代表臨床準確率。

## 停止條件與結論

事前通過條件為四張藥品皆正確類別一框，空背景零框。由於 drug004 仍漏辨，**未達通過條件，停止本輪，不進行 Pi 重跑**，不再更改裁切範圍或門檻，亦未重新訓練。

本次證據支持「相同五張影像經固定中央取景後，部分預測改善」，不能歸納為已可靠支援新相機。中央取景同時改變目標在模型輸入中的尺度、可見背景及補邊幾何；不是只有尺度被單獨控制的實驗，也沒有新增獨立樣本。顏色可由人眼辨認，不代表偵測模型必然能在不同取像條件下找到物件；零框不等於把黃色或綠色分類成白色。模型依賴哪些特徵、膠囊漏辨原因仍待後續研究。

考量 10/28 書審截止，本輪結果應作為限制與錯誤分析紀錄，下一步先整理書審已完成成果，不以繼續試參數換取展示成功。原手機照片上的 Pi 部署與數量比對成果維持原適用範圍，不挪用為新相機成果。

## 重現指令

以下輸出均已存在，程式會拒絕覆寫；若需重現，另取版本名稱，但本輪不再變更條件或追加參數試驗。

```bash
.venv/bin/python scripts/prepare_camera_roi.py \
  --images data/raw/camera-v3-燕麥背景 \
  --cases data/splits/camera-v3-oatmeal01.json \
  --output data/derived/camera-v3-oatmeal01-roi2x \
  --record outputs/camera-v3-oatmeal01-roi2x-preparation-v001/manifest.json

.venv/bin/python scripts/evaluate_camera_minimal.py \
  --model outputs/baseline-20261016-150812-917344/weights/best.pt \
  --images data/derived/camera-v3-oatmeal01-roi2x \
  --cases data/splits/camera-v3-oatmeal01-roi2x.json \
  --output outputs/camera-v3-oatmeal01-roi2x-mac-cpu-conf025-v001
```

推論輸出 `result.json` 含完整設定、環境、衍生圖雜湊、程式雜湊及預測座標。推論程式只更新限制文字以區分輸入衍生圖與其內部未另裁切，不改推論設定。本次尚未提交 Git 或推送 GitHub；原圖、衍生圖、權重及 outputs 須私人備份。
