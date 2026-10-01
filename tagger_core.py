"""
ONNX 推論 + 検閲判定のコア部分。ComfyUI ノードから呼ばれる。

* wd-eva02-large-tagger-v3 (FP16 ONNX) で画像をタグ化
* `uncensored` / `censored` / `no_explicit` / `unknown` の 4 値判定
* タグ文字列も同時に返す (ComfyUI-WD14-Tagger と同じフォーマット)

モデル探索順:
    1. ComfyUI の `models/wd14_tagger/` (folder_paths から取得)
    2. このパッケージ直下の `models/` フォルダ
    3. 上記のいずれにも無ければ HuggingFace から FP32 を落として FP16 化
"""
from __future__ import annotations

import csv
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image


if sys.platform == "win32":
    ort.preload_dlls(directory=str(Path(torch.__file__).resolve().parent / "lib"))
elif sys.platform.startswith("linux"):
    ort.preload_dlls()


# ---- 検閲判定に使うタグ ----
CENSOR_TAGS: tuple[str, ...] = (
    "censored", "mosaic_censoring", "bar_censor", "convenient_censoring",
    "heart_censor", "pointless_censoring", "out-of-frame_censoring",
    "hair_censor", "novelty_censor", "blur_censor", "steam_censor",
    "censored_nipples", "identity_censor", "tail_censor", "blank_censor",
    "light_censor", "soap_censor", "character_censor", "censored_text",
    "transparent_censoring",
)
UNCENSORED_TAG = "uncensored"
GENITAL_TAGS: tuple[str, ...] = ("pussy", "penis")

MODEL_BASENAME = "wd-eva02-large-tagger-v3-fp16.onnx"
CSV_BASENAME = "wd-eva02-large-tagger-v3.csv"
HF_BASE = "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main"
HF_ONNX_URL = f"{HF_BASE}/model.onnx?download=true"
HF_CSV_URL = f"{HF_BASE}/selected_tags.csv?download=true"

PACKAGE_DIR = Path(__file__).resolve().parent
LOCAL_MODELS_DIR = PACKAGE_DIR / "models"

CensorLabel = Literal["censored", "uncensored", "no_explicit", "unknown"]


@dataclass
class TagResult:
    """1 枚分の結果。"""
    label: CensorLabel
    tag_string: str           # ComfyUI-WD14-Tagger と同じ "tag1, tag2" 形式 (括弧はエスケープ)
    uncensored_score: float
    max_censor_score: float
    top_censor_tag: str
    genital_score: float
    top_genital_tag: str
    inference_ms: float = 0.0
    raw_tags: list[tuple[str, float]] = field(default_factory=list)


# ---- モデル探索 / 自動ダウンロード ----

def _comfy_models_dir() -> Path | None:
    """ComfyUI の `models/wd14_tagger/` ディレクトリを取得 (見つからなければ None)。"""
    try:
        import folder_paths  # type: ignore
    except ImportError:
        return None
    if "wd14_tagger" in getattr(folder_paths, "folder_names_and_paths", {}):
        try:
            paths = folder_paths.get_folder_paths("wd14_tagger")
            if paths:
                d = Path(paths[0])
                d.mkdir(parents=True, exist_ok=True)
                return d
        except Exception:
            pass
    # 既知 wd14_tagger フォルダが未登録なら標準の `models/` 直下に作る
    base = getattr(folder_paths, "models_dir", None)
    if base:
        d = Path(base) / "wd14_tagger"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return None


def _download(url: str, dest: Path, label: str | None = None) -> None:
    label = label or dest.name
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[onnxtagger] downloading {label}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "comfyui-onnxtagger/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1024 * 256
        written = 0
        t0 = time.perf_counter()
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                written += len(buf)
                if total:
                    pct = written / total * 100
                    sys.stdout.write(f"\r  {written / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
                    sys.stdout.flush()
        sys.stdout.write("\n")
    tmp.rename(dest)


def _ensure_fp16_model(target_dir: Path) -> tuple[Path, Path]:
    """target_dir に FP16 モデルと CSV があることを保証して、両方のパスを返す。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    fp16 = target_dir / MODEL_BASENAME
    csv_path = target_dir / CSV_BASENAME

    if not csv_path.exists():
        _download(HF_CSV_URL, csv_path, label="selected_tags.csv")

    if not fp16.exists():
        # FP32 を一旦落として FP16 に変換
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError as e:
            raise RuntimeError(
                "FP16 変換に onnx / onnxconverter_common が必要です。"
                "`pip install onnx onnxconverter_common` を実行してください。"
            ) from e

        fp32 = target_dir / "wd-eva02-large-tagger-v3.onnx"
        if not fp32.exists():
            _download(HF_ONNX_URL, fp32, label="model.onnx (FP32, 約 1.2 GB)")

        print(f"[onnxtagger] converting to FP16 -> {fp16.name}")
        model = onnx.load(str(fp32))
        fp16_model = float16.convert_float_to_float16(
            model,
            op_block_list=["LayerNormalization", "Softmax", "Sigmoid",
                           "ReduceMean", "Div", "Conv"],
            keep_io_types=True,
            disable_shape_infer=True,
        )
        tmp = fp16.with_suffix(fp16.suffix + ".part")
        onnx.save(fp16_model, str(tmp))
        tmp.rename(fp16)
        try:
            fp32.unlink()
        except OSError:
            pass

    return fp16, csv_path


def resolve_model_paths(prefer_dir: Path | str | None = None) -> tuple[Path, Path]:
    """FP16 ONNX と CSV のパスを解決する。無ければダウンロード + 変換する。

    探索順:
        1. prefer_dir (引数で指定)
        2. ComfyUI の `models/wd14_tagger/`
        3. 本パッケージの `models/`
    どれにもなければ ComfyUI フォルダを優先して、無ければパッケージフォルダにダウンロード。
    """
    candidates: list[Path] = []
    if prefer_dir is not None:
        candidates.append(Path(prefer_dir))
    comfy_dir = _comfy_models_dir()
    if comfy_dir is not None:
        candidates.append(comfy_dir)
    candidates.append(LOCAL_MODELS_DIR)

    for d in candidates:
        fp16 = d / MODEL_BASENAME
        csv = d / CSV_BASENAME
        if fp16.exists() and csv.exists():
            return fp16, csv

    # 何処にも無ければ、書き込み可能な最初の候補にダウンロード
    target = comfy_dir or LOCAL_MODELS_DIR
    return _ensure_fp16_model(target)


# ---- 推論 / 判定 ----

class OnnxTagger:
    """ONNX セッション + 判定ロジックをまとめたインスタンス。

    ComfyUI ノードからは「グローバル単一インスタンス」として使い回す想定。
    """

    def __init__(
        self,
        model_path: Path | None = None,
        csv_path: Path | None = None,
        providers: Iterable[str] | None = None,
    ) -> None:
        if model_path is None or csv_path is None:
            model_path, csv_path = resolve_model_paths()
        self.model_path = model_path
        self.csv_path = csv_path

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = int(self.session.get_inputs()[0].shape[1])

        self.tags, self.general_index, self.character_index = self._load_tags(csv_path)
        idx = {t: i for i, t in enumerate(self.tags)}
        self._uncensored_idx = idx[UNCENSORED_TAG]
        self._censor_indices = [idx[t] for t in CENSOR_TAGS if t in idx]
        self._genital_indices = [idx[t] for t in GENITAL_TAGS if t in idx]

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    @staticmethod
    def _load_tags(csv_path: Path) -> tuple[list[str], int, int]:
        tags: list[str] = []
        general_index: int | None = None
        character_index: int | None = None
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if general_index is None and row[2] == "0":
                    general_index = reader.line_num - 2
                elif character_index is None and row[2] == "4":
                    character_index = reader.line_num - 2
                tags.append(row[1])
        if general_index is None:
            general_index = 0
        if character_index is None:
            character_index = len(tags)
        return tags, general_index, character_index

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        size = self.input_size
        ratio = float(size) / max(image.size)
        new_size = tuple(int(x * ratio) for x in image.size)
        image = image.resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (size, size), (255, 255, 255))
        square.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
        arr = np.array(square).astype(np.float32)
        arr = arr[:, :, ::-1]                # RGB -> BGR
        arr = np.expand_dims(arr, 0)
        return arr

    def tag(
        self,
        image: Image.Image,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        uncensored_threshold: float = 0.10,
        censored_threshold: float = 0.50,
        genital_threshold: float = 0.20,
        replace_underscore: bool = False,
        trailing_comma: bool = False,
        exclude_tags: str = "",
    ) -> TagResult:
        x = self._preprocess(image)
        t0 = time.perf_counter()
        probs = self.session.run([self.output_name], {self.input_name: x})[0][0]
        t1 = time.perf_counter()

        # ---- 検閲判定 (4 値) ----
        u_score = float(probs[self._uncensored_idx])
        c_idx_local = int(np.argmax(probs[self._censor_indices]))
        c_score = float(probs[self._censor_indices[c_idx_local]])
        c_name = CENSOR_TAGS[c_idx_local]
        g_idx_local = int(np.argmax(probs[self._genital_indices]))
        g_score = float(probs[self._genital_indices[g_idx_local]])
        g_name = GENITAL_TAGS[g_idx_local]

        if u_score >= uncensored_threshold:
            label: CensorLabel = "uncensored"
        elif c_score >= censored_threshold:
            label = "censored"
        elif g_score < genital_threshold:
            label = "no_explicit"
        else:
            label = "unknown"

        # ---- タグ文字列 (WD14Tagger と同じ規則) ----
        general = [(self.tags[i], float(probs[i]))
                   for i in range(self.general_index, self.character_index)
                   if probs[i] > threshold]
        character = [(self.tags[i], float(probs[i]))
                     for i in range(self.character_index, len(self.tags))
                     if probs[i] > character_threshold]

        if replace_underscore:
            general = [(t.replace("_", " "), s) for t, s in general]
            character = [(t.replace("_", " "), s) for t, s in character]

        all_tags = character + general
        excluded = {s.strip() for s in exclude_tags.lower().split(",") if s.strip()}
        all_tags = [t for t in all_tags if t[0].lower() not in excluded]

        # 括弧をエスケープ (Stable Diffusion プロンプトで強調と誤認されるため)
        formatted = [t[0].replace("(", r"\(").replace(")", r"\)") for t in all_tags]
        sep = "," if trailing_comma else ", "
        tag_string = sep.join(formatted)
        if trailing_comma and tag_string:
            tag_string = tag_string + ","

        return TagResult(
            label=label,
            tag_string=tag_string,
            uncensored_score=u_score,
            max_censor_score=c_score,
            top_censor_tag=c_name,
            genital_score=g_score,
            top_genital_tag=g_name,
            inference_ms=(t1 - t0) * 1000.0,
            raw_tags=all_tags,
        )


# ---- グローバル インスタンス キャッシュ ----
# ComfyUI ノードはユーザ操作の度にインスタンス化されることがあるので、
# モデルの重複ロード (1.2 GB) を避けるためグローバル キャッシュする。
_GLOBAL_TAGGER: OnnxTagger | None = None
_GLOBAL_TAGGER_KEY: tuple[str, str] | None = None


def get_or_create_tagger(
    model_path: Path | None = None,
    csv_path: Path | None = None,
    providers: Iterable[str] | None = None,
) -> OnnxTagger:
    global _GLOBAL_TAGGER, _GLOBAL_TAGGER_KEY
    if model_path is None or csv_path is None:
        model_path, csv_path = resolve_model_paths()
    key = (str(model_path), str(csv_path))
    if _GLOBAL_TAGGER is None or _GLOBAL_TAGGER_KEY != key:
        _GLOBAL_TAGGER = OnnxTagger(model_path=model_path, csv_path=csv_path, providers=providers)
        _GLOBAL_TAGGER_KEY = key
    return _GLOBAL_TAGGER
