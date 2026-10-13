# 環境設定與版本紀錄

本機預設：Python 3.14.7（2026-10-13 查得）。尚未安裝訓練套件。

建議先選 Python 3.11 的最新修補版本建立獨立環境；正式安裝前確認 PyTorch、Ultralytics 對作業系統與處理器的支援。Python 3.11 已處於安全維護階段，選它是為了套件相容性，並非因它是最新主版本。

安裝相容的 Python 3.11 後，在專案目錄執行：

```sh
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ultralytics
python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__)"
python -m pip freeze > requirements-mac.lock.txt
python --version > python-version-mac.txt
```

上面是待操作指令，尚未執行。首次成功安裝後鎖定實際版本，不憑空填套件版本。往後重建相同平台環境使用 `python -m pip install -r requirements-mac.lock.txt`。樹莓派使用另一个環境與 requirements-pi.lock.txt，需確認 ARM 套件可用性。

實驗另記：作業系統、處理器、使用 CPU/MPS/CUDA、套件版本、模型來源與版本、影像尺寸、epochs、batch、seed、資料切分版本與 Git commit。版本鎖定不等於保證完全相同數值。
