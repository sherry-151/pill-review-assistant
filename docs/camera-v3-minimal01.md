# Camera Module 3 最小辨識測試

日期：2026-10-25。新相機照片已分別在 **Mac CPU 與 Raspberry Pi 5 CPU** 完成離線測試與同圖比對；不是相機即時串接或動態出藥驗證。

## 資料與保存

依使用者提供的實際擺放與檔名，drug001 至 drug004 各單顆一張，另有空背景一張。五張從 `data/raw/` 搬至 `data/raw/camera-v3-minimal01/`；原檔名及位元內容不變，搬移前後 SHA-256 一致。檔名中的 `v2` 是使用者原命名，不代表模型已重新訓練或使用第二版權重。

五張均為 4608 × 2592、EXIF orientation 1。推論前程式仍統一執行 EXIF 方向校正，不裁切、不增強、不另外放大目標。原圖與失敗案例全部保存；沒有用這五張訓練或調整門檻。此批是探索性開發測試，不是盲測。

## 使用者補充的拍攝差異

|條件|原手機訓練照片|本批樹莓派新相機照片|
|---|---|---|
|拍攝倍率|手機介面 2×|正常拍攝，沒有 2×|
|距離|15 cm|15 cm|
|背景紙|燕麥色|灰色|

以上為使用者自述，尚未以尺規或色卡量測。相同距離不等於相同視野或藥錠像素大小；手機 2× 亦不能單憑名稱判斷是光學或數位倍率。兩種相機本身的視野也可能不同，不能直接推定藥錠影像恰好相差兩倍。背景與目標尺度均可能形成輸入分布差異，但目前同時改變多個條件，不能斷定哪個條件造成辨錯或漏辨。先保留本次整張原圖的固定設定結果；若後續比較背景或取景範圍，須另建批次並記錄控制變因，不能覆蓋本次結果。

## 凍結設定

- 權重：`outputs/baseline-20261016-150812-917344/weights/best.pt`
- SHA-256：`d195469586c0511b28c5052735372e609d5003c5923c3fb23ec207d22055a0f4`
- conf 0.25、imgsz 640、NMS IoU 0.7、CPU FP32、max_det 300、rect True；augment False、agnostic_nms False。
- 使用目前套件的 `quantize=None`，未傳入已棄用的 `half` 參數。未量化或重新訓練。
- Mac：Python 3.11.17、torch 2.14.1、torchvision 0.29.1、ultralytics 8.4.177、Pillow 12.3.0。

## 逐張結果與人工檢視

|照片|實際擺放|模型輸出|判讀|
|---|---|---|---|
|v2-drug001.jpg|drug001 一顆|drug002 一框，信心值 0.980921|框在藥錠上，但辨錯類別；不是完全沒有偵測框|
|v2-drug002.jpg|drug002 一顆|drug002 一框，信心值 0.847239|類別及數量符合，框包含目標但上下有背景餘裕|
|v2-drug003.jpg|drug003 一顆|零框|在固定門檻下漏辨|
|v2-drug004.jpg|drug004 一顆|零框|在固定門檻下漏辨|
|v2-空白.jpg|零顆|零框|本張未出現背景誤報|

逐張查看全部五張輸出影像，沒有多框案例。四張有藥照片僅一張品項與數量符合；合併空背景則為五張中的兩張符合。這是個案計數，不宣稱臨床準確率，也不把空背景混進藥品辨識成功率。處理錯誤為零。

沒有人工定位框，因此不計算定位 IoU、precision/recall 或 mAP。人工目視僅確認框是否在目標附近，不代表已通過正式定位評估。信心值不是校準後的正確機率；drug001 的高信心錯類別尤其不能以「分數高」代替人工確認。

## 結論

原本用手機照片訓練的模型可以載入並對新相機照片執行推論，但在本批原圖及固定設定下，**未通過直接沿用的最小辨識檢查**：出現一張辨錯及兩張漏辨。因此目前不能宣稱已可靠適用於新相機，亦不能宣稱完成 Pi 相機辨識與數量比對示範。

這不是相機硬體故障的證據，也不能僅憑五張照片確定失敗原因。目標尺度、光線、白平衡、背景、姿態及訓練資料涵蓋程度都可能影響輸出；尚未進行控制變因實驗，不能把失敗歸因於其中一項。相機照片看起來較清楚，不保證模型能辨識。

同五張、同權重與設定的 Pi CPU 重跑與比對已完成，結果見下一節。後續若另做取像或資料調整，使用新批次與新輸出版本，保留本批失敗基準，不覆寫或挑掉失敗照片。新增的開發嘗試不得混算為獨立測試成果。

## Pi CPU 重跑與 Mac 同圖核對

2026-10-25 已下載 Pi 完整結果，對照 Mac v002 基準完成核對。

- Mac 基準：`outputs/camera-v3-minimal01-mac-cpu-conf025-v002/result.json`
- Pi 結果：`outputs/camera-v3-minimal01-pi-cpu-conf025-v001/result.json`
- Pi 結果檔 SHA-256：`60ef4a45389a5e7a22df369f8d532eb521b17e8a2ac374d422d78dc282eaadfc`
- 兩端權重雜湊、五張原圖雜湊、程式雜湊及完整推論設定均一致。程式雜湊為 `a3387169a0bd8ab6b4a1bee81d318ef69399d768c6cb00da75af5d413866860e`。
- Pi 環境：Python 3.13.5、Linux 6.18.50+rpt-rpi-2712 aarch64、torch 2.14.1+cpu、torchvision 0.29.1+cpu、ultralytics 8.4.177、Pillow 12.3.0。與 Mac 的平台及 Python 不同，不主張環境完全相同。

|照片|Pi 輸出|與 Mac 比較|
|---|---|---|
|v2-drug001.jpg|drug002 一框，0.980920553|兩端均辨錯成 drug002|
|v2-drug002.jpg|drug002 一框，0.847237587|兩端均符合類別及數量|
|v2-drug003.jpg|零框|兩端均漏辨|
|v2-drug004.jpg|零框|兩端均漏辨|
|v2-空白.jpg|零框|兩端本張均無背景誤報|

五張的類別與數量全部跨裝置一致，但不代表五張皆辨識正確。有藥四張僅一張品項及數量符合；空背景一張另計。兩端均無多框，Pi 處理錯誤為零，五張輸出 PNG 與五份標籤均存在，零框案例未省略。兩個有框案例的最大信心值絕對差為 `1.1324882507324219e-6`，原圖像素座標最大絕對差為 `0.000244140625`，並非逐位完全相同。這是跨裝置預測框差異，不是與人工定位框的正確率比較。

本次辨錯與漏辨在兩端均可重現，因此不是只有切換至 Pi 才出現的錯誤。結果支持此批固定輸入的跨裝置輸出一致性，不支持模型已可靠適用新相機，也不能據此確定失敗原因。下一步可另建固定模型的拍攝條件對照，逐項檢查背景、目標尺度等差異，不能混入本批或刪除本批失敗照片。

## 重現與輸出

在 Mac 專案目錄執行（下列輸出名稱若已存在，程式會拒絕覆寫；重跑需改用新名稱）：

```bash
.venv/bin/python scripts/evaluate_camera_minimal.py \
  --model outputs/baseline-20261016-150812-917344/weights/best.pt \
  --images data/raw/camera-v3-minimal01 \
  --output outputs/camera-v3-minimal01-mac-cpu-conf025-v002
```

完整輸出：`outputs/camera-v3-minimal01-mac-cpu-conf025-v001/`。`annotated/` 保存全部五張 PNG，包括零框影像；`labels/` 保存五份預測標籤，零框使用明確空檔；`result.json` 保存設定、環境、完整分數、座標、輸入與程式雜湊。人工逐張檢視紀錄即本文件。

為使同一程式可部署至 Pi，後續只將輸出中寫死的 Mac 主機說明改為實際平台與主機名稱；推論設定不變。原 v001 程式另存於其輸出資料夾，原結果不覆寫。修正後的 Mac 重跑結果存於 `outputs/camera-v3-minimal01-mac-cpu-conf025-v002/`，作為相同程式的跨裝置比較基準；其預測與 Mac v001 完全相同，Pi 實測比較如上。

## 在 Raspberry Pi CPU 重跑

Mac 本機逐行執行，下列 `sherry.local` 是先前已成功連線的主機名稱。若無法連線，先確認 Pi 已開機並在相同網路；不沿用可能已變更的舊 IP。首次連線仍須核對主機身分。

```bash
ssh sherry@sherry.local 'mkdir -p /home/sherry/pill-review-pi/camera-v3-minimal01'
scp /Users/sherry/projects/pill-review-assistant/data/raw/camera-v3-minimal01/*.jpg sherry@sherry.local:/home/sherry/pill-review-pi/camera-v3-minimal01/
scp /Users/sherry/projects/pill-review-assistant/scripts/evaluate_camera_minimal.py sherry@sherry.local:/home/sherry/pill-review-pi/
ssh sherry@sherry.local
```

Pi SSH 視窗逐步執行；只啟用既有環境，不更新套件：

```bash
source /home/sherry/pill-inference-venv/bin/activate
sha256sum /home/sherry/pill-review-pi/models/best.pt
python /home/sherry/pill-review-pi/evaluate_camera_minimal.py \
  --model /home/sherry/pill-review-pi/models/best.pt \
  --images /home/sherry/pill-review-pi/camera-v3-minimal01 \
  --output /home/sherry/pill-review-pi/outputs/camera-v3-minimal01-pi-cpu-conf025-v001
```

權重雜湊必須等於上方凍結值，程式也會檢查；不符時停止，不替換成其他模型。輸出已存在時不刪除，改用新版本名稱。完成後另開 Mac 本機終端機下載：

```bash
scp -r sherry@sherry.local:/home/sherry/pill-review-pi/outputs/camera-v3-minimal01-pi-cpu-conf025-v001 /Users/sherry/projects/pill-review-assistant/outputs/
```

下載後比對兩端 `result.json` 的 `model_sha256`、`script_sha256`、`input_sha256`、`settings`；確認一致後才比較逐張類別、數量、信心值及框座標。Python、套件與平台版本另行記錄，不要求兩端數值逐位完全相同。此流程重跑既有五張，不重新拍攝、不使用 Pi 相機當下影像，也不訓練或調門檻。

測試程式與首版紀錄已由 `f4a37ba` 保存並推送；本次補記 Pi 同圖核對結論，公開內容限程式與文字紀錄。原圖、權重及 outputs 被 Git 忽略，須另外私人備份，不隨本次推送上傳。
