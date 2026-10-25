# 新相機中央取景：conf 0.10 候選實測

日期：2026-10-25。依使用者指定，只新增 conf 0.10 這一組候選；沒有執行 conf 0.15、imgsz 960／1280、重訓或 Pi 推論。這五張已參與開發，不是獨立驗證資料。

## 固定條件

- 原 best.pt SHA-256：`d195469586c0511b28c5052735372e609d5003c5923c3fb23ec207d22055a0f4`。
- 同五張燕麥背景原圖的既有中央取景衍生圖，座標 `[1152,648,3456,1944]`，尺寸 2304 × 1296；無個別取景或增強。程式核對照片及準備紀錄雜湊。
- imgsz 640、conf **0.10**、NMS IoU 0.7、Mac CPU FP32、max_det 300、rect True、augment False、agnostic_nms False。相對中央取景基準只改 conf。
- Python 3.11.17、torch 2.14.1、torchvision 0.29.1、Ultralytics 8.4.177、Pillow 12.3.0。
- 原 conf 0.25 固定測試程式與結果不變；候選另建入口程式，共用完整輸出核心，全部類別均保存。

## 全部五張結果

|實際案例|conf 0.25 中央取景基準|conf 0.10 候選|候選保留框分數|
|---|---|---|---|
|drug001 一顆|drug001 一框|drug001 一框|0.964250505|
|drug002 一顆|drug002 一框|drug002 一框|0.694553792|
|drug003 一顆|drug003 一框|drug003 一框|0.872150600|
|drug004 一顆|零框，漏辨|drug004 一框|0.100052796|
|空背景|零框|零框|無|

候選處理錯誤零、達 max_det 影像零。已逐張目視檢查五張全部類別標註圖：四個框分別位於對應藥品上，膠囊框大致包含完整膠囊；未見錯類別或多框。沒有人工框真值，不報定位 IoU 或 mAP。

結論僅為「本批五張開發案例的品項／數量皆符合，且人工查看框位置合理」。不得稱為新相機準確率 100%、跨相機泛化成功或臨床驗證。降低 conf 只是保留原本低分框，不代表模型能力已改善。

膠囊分數僅比 0.10 高約 **0.0000528**，非常接近門檻；姿態、照明、重新拍攝或運算平台的細微差異都可能影響是否保留，尚無穩定性證據。模型分數不是辨識正確機率。

## 保存與重現

輸出：`outputs/camera-v3-demo-dev01/candidate-conf010-img640-v001/`。內含五張 `annotated-all/`、全部類別 `labels/`（含空標籤）、逐框及環境紀錄 `result.json`。`drug004-only-display/` 僅供查看，判斷以全部類別輸出為準。既有失敗基準與低門檻診斷全部保留。

```bash
.venv/bin/python scripts/evaluate_camera_candidate_conf010.py \
  --model outputs/baseline-20261016-150812-917344/weights/best.pt \
  --images data/derived/camera-v3-oatmeal01-roi2x \
  --cases data/splits/camera-v3-oatmeal01-roi2x.json \
  --preparation outputs/camera-v3-oatmeal01-roi2x-preparation-v001/manifest.json \
  --output outputs/camera-v3-demo-dev01/candidate-conf010-img640-v001
```

既有輸出拒絕覆寫。JSON 另保存每張輸入、案例設定、準備紀錄、入口程式與共用核心的雜湊。此次入口 SHA-256 為 `1d83ac72d38f3e7f4c4d4cd9fcce8fb87f144bf06b6429bcacb0e794d3e6331d`；核心為 `f52a850bd9149d5488051b0f6ef4d91c582dcf8689fd333ece742495cdabb8a1`。

七項既有案例判斷單元測試通過。圖像、權重及 outputs 不納入 Git，需另行私人備份；Git 保存程式、案例設定與文字紀錄。

## 下一步與停止界線

本次到此停止，不再跑其他候選。暫凍結中央取景＋conf 0.10／imgsz 640／IoU 0.7／CPU，依[展示開發計畫](camera-v3-demo-development-plan-v1.md)新拍四類各三張及空背景三張，共十五張，拍攝前記錄真實品項與數量，不依預測挑照片。此批尚未拍攝或評估。

新照片若失敗，保留失敗，不能臨時改參數後仍稱獨立確認。之後才核對相同資料在 Pi CPU 的輸出與耗時，最後測多顆預期數量比對；以上均未在本次完成。書審只填已完成部分，10/28 上傳優先。
