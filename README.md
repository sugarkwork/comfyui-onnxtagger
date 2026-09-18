# comfyui-onnxtagger

ComfyUI のカスタムノード。ONNX (FP16 + CUDA) で `wd-eva02-large-tagger-v3` を呼び、
**タグ文字列 + 検閲状態 4 値 (BOOLEAN)** をひとつのノードから出力する。

## ノード

`ONNX Tagger (Censor Detector)` (内部 ID `OnnxTagger|onnxtagger`)

### 入力

| 名前 | 型 | 既定 | 説明 |
|---|---|---|---|
| `image` | IMAGE | - | バッチ画像 (B, H, W, C) |
| `threshold` | FLOAT | 0.35 | 一般タグのしきい値 |
| `character_threshold` | FLOAT | 0.85 | キャラ タグのしきい値 |
| `uncensored_threshold` | FLOAT | 0.10 | uncensored 判定のしきい値 |
| `censored_threshold` | FLOAT | 0.50 | censored 判定のしきい値 |
| `genital_threshold` | FLOAT | 0.20 | no_explicit 判定のしきい値 |
| `replace_underscore` | BOOLEAN | False | タグの `_` を空白に置換 |
| `trailing_comma` | BOOLEAN | False | 末尾にカンマを付ける |
| `exclude_tags` | STRING | "" | 除外タグ (カンマ区切り) |

### 出力 (5 つ)

| 名前 | 型 | 内容 |
|---|---|---|
| `tags` | STRING | `"tag1, tag2, ..."` 形式のタグ文字列 (ComfyUI-WD14-Tagger と互換) |
| `is_uncensored` | BOOLEAN | 無修正と判定 |
| `is_censored` | BOOLEAN | 検閲済みと判定 |
| `is_no_explicit` | BOOLEAN | 性器が描かれていないと判定 |
| `is_unknown` | BOOLEAN | 上記いずれでもない (要・人手確認) |

4 値判定の優先順位:

1. `uncensored タグ ≥ uncensored_threshold` → **is_uncensored**
2. `max(検閲タグ群) ≥ censored_threshold` → **is_censored**
3. `max(pussy, penis) < genital_threshold` → **is_no_explicit**
4. それ以外 → **is_unknown**

検閲タグ群: `censored`, `mosaic_censoring`, `bar_censor`, `convenient_censoring`,
`heart_censor`, `pointless_censoring`, `out-of-frame_censoring`, `hair_censor`,
`novelty_censor`, `blur_censor`, `steam_censor`, `censored_nipples`, `identity_censor`,
`tail_censor`, `blank_censor`, `light_censor`, `soap_censor`, `character_censor`,
`censored_text`, `transparent_censoring`

しきい値の根拠は親プロジェクトの `report_thresholds.md` (249 枚での検証) を参照。

## モデルファイル

ノードは初回実行時に以下を **自動でダウンロード + FP16 変換** する:

- `wd-eva02-large-tagger-v3-fp16.onnx` (FP16, 約 600 MB)
- `wd-eva02-large-tagger-v3.csv` (タグ定義)

保存先の優先順位:

1. ComfyUI の `models/wd14_tagger/` (既存の WD14 Tagger プラグインと共存可能)
2. このパッケージ直下の `models/`

既に上記のいずれかに FP16 ONNX が置かれていればそれを使う。

## インストール

ComfyUI の `custom_nodes/` 直下にこのフォルダを配置:

```
ComfyUI/custom_nodes/comfyui-onnxtagger/
```

依存パッケージ:

```bash
pip install -r requirements.txt
```

`onnxruntime-gpu` 用に CUDA 12 / cuDNN 9 が必要。pip wheel で揃うので OS 側へのインストールは不要:

- Linux: `nvidia-cudnn-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12` を pip で入れる
- Windows: 同上 (Windows wheel は DLL を提供)

`tagger_core.py` が import 時に `os.add_dll_directory()` (Windows) / `ctypes.CDLL` プリロード (Linux)
で pip wheel 版 CUDA libs を自動的に検索パスに加えるので、`LD_LIBRARY_PATH` / `PATH` を
手動でいじる必要は通常ない。

## ファイル構成

```
comfyui-onnxtagger/
├── __init__.py            # ComfyUI への登録 (NODE_CLASS_MAPPINGS)
├── nodes.py               # OnnxTagger ノード本体
├── tagger_core.py         # ONNX 推論 + 判定 + モデル自動セットアップ
├── requirements.txt
├── pyproject.toml
├── README.md
└── models/                 # 初回実行時に自動的に埋まる (空でも可)
```

## ライセンス

- モデル: [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) (Apache 2.0)
- 本ノード コード: MIT
