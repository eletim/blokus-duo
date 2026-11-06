# gen3公開統合検証
基点origin/main6be347f。研究全体mergeなし。必要ファイルだけ追加。

- Bは67482585のB64/LR3e-4/5120/1751401,2,3。16e3baa models.jsonと3推論ファイルのSHA完全一致。manifestへ元のpath/commit/SHAを保存。公開推論は研究module/履歴/大型indexを参照しない。
- updateは640bd3f由来の同期集約・常駐入力と67482585までのminibatch引数のみ。raggedモジュール/フラグなし。loss・Adam・float32・KL停止は維持、16→64は学習条件変更。
- gen3既定1751401（成績選択なし）、旧gen2/2048/1401-3維持。gen3で1401-3は対応seedの別名。旧global AI defaultファイルは差分なし。
- 新cloneのCPU-only torch環境: 76passed/4CUDA skipped。CUDA環境: 80passed。旧6ch oracle testはgen2を明示、新規gen3 CPU/CUDA出力・value・greedy・SHA検証を追加。
- 新cloneでgen3/gen2/2048×CPU-only/CUDA×AI対戦/人間先手/後手の18経路PASS。人間は合法手0を入力して次入力でEOF終了を確認。CPU native未build auto fallbackとCUDA native buildも確認。
- CLI起動/モデル初期化を含むgen3対gen2の1対局wall CPU0.915s/CUDA1.535s。これは起動込みsmokeで推論速度比較ではない。確認試験のGPU/native手あたり費用はconfirmation-audit.json。後続ragged速度は未採用・別成果。
- CPU/CUDAの4ch/64/LR3e-4学習4iteration vs2+別process再開2はmodel/past/optimizer/全RNG bit一致（past交換4）。省略設定の継承、変更拒否、明示変更履歴、旧metadataなしcheckpointの16/optimizer LR継承PASS。Path型resumeの設定JSONエラーを追加テストで発見し修正。
- 本学習/評価の再実行なし。短い学習とテストのみ。公開Bの強さ根拠は固定256prefix試験でありCLI smoke勝敗は棋力根拠にしない。

再現:
```bash
python -m pytest -q
python scripts/check_gen3_resume.py
# 新規clone先、CPU-only Python、CUDA Pythonを渡す
python scripts/check_gen3_clone.py /path/to/fresh-clone /path/to/cpu-python /path/to/cuda-python
```
元の研究記録: research/ppo-lr-64/artifacts/ppo-lr64、research/ppo-fixed-confirmation/artifacts/ppo-confirmation。mainには要約/監査と推論用重みのみ掲載。
