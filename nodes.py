"""
ComfyUI カスタムノード:

`OnnxTagger | onnxtagger`

入力:
    image (IMAGE)              : 1 枚 〜 N 枚のバッチ画像
    threshold (FLOAT)          : 一般タグのしきい値 (既定 0.35)
    character_threshold (FLOAT): キャラタグのしきい値 (既定 0.85)
    uncensored_threshold (FLOAT): uncensored 判定 (既定 0.10)
    censored_threshold (FLOAT)  : censored 判定 (既定 0.50)
    genital_threshold (FLOAT)   : no_explicit 判定 (既定 0.20)
    replace_underscore (BOOL)
    trailing_comma (BOOL)
    exclude_tags (STRING)

出力 (5 つ):
    tags (STRING)         : "tag1, tag2, ..." 形式のタグ文字列
    is_uncensored (BOOL)  : 無修正と判定された
    is_censored   (BOOL)  : 検閲済みと判定された
    is_no_explicit(BOOL)  : 性器が描かれていない
    is_unknown    (BOOL)  : 上記いずれでもない (要確認)

OUTPUT_IS_LIST = True にしているため、バッチ画像の場合はそれぞれ N 個のリストを返す。
バッチ サイズ 1 の通常運用ではダウンストリーム側で `[0]` 取り出しが必要になるが、
ComfyUI 側で自動的にスカラー化されるノード結合パターンが多いため、多くの場合そのまま使える。
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

try:
    import comfy.utils  # type: ignore
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

from .tagger_core import TagResult, get_or_create_tagger


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI の IMAGE テンソル (H, W, C, 値域 [0, 1]) を PIL に変換。"""
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]  # 念のため B 次元が残っていたら除去
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class OnnxTagger:
    """ONNX で画像をタグ化 + 検閲状態 (4 値) を判定する ComfyUI カスタムノード。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "character_threshold": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "uncensored_threshold": ("FLOAT", {
                    "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "uncensored タグがこの値以上で is_uncensored=True",
                }),
                "censored_threshold": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "検閲タグ群の最大値がこの値以上で is_censored=True (uncensored でない場合)",
                }),
                "genital_threshold": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "max(pussy, penis) がこの値未満で is_no_explicit=True (上記いずれでもない場合)",
                }),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("tags", "is_uncensored", "is_censored", "is_no_explicit", "is_unknown")
    OUTPUT_IS_LIST = (True, True, True, True, True)
    FUNCTION = "tag"
    CATEGORY = "image/tagging"
    OUTPUT_NODE = True

    def tag(
        self,
        image: torch.Tensor,
        threshold: float,
        character_threshold: float,
        uncensored_threshold: float,
        censored_threshold: float,
        genital_threshold: float,
        replace_underscore: bool,
        trailing_comma: bool,
        exclude_tags: str,
    ):
        tagger = get_or_create_tagger()

        # IMAGE は (B, H, W, C) のテンソル
        if image.ndim == 3:  # 念のため (H, W, C) も許容
            image = image.unsqueeze(0)
        batch_size = int(image.shape[0])

        if HAS_COMFY:
            pbar = comfy.utils.ProgressBar(batch_size)
        else:
            pbar = None

        tag_strings: list[str] = []
        flags_uncensored: list[bool] = []
        flags_censored: list[bool] = []
        flags_no_explicit: list[bool] = []
        flags_unknown: list[bool] = []

        ui_tags: list[str] = []  # ComfyUI の preview に出すための補助
        ui_labels: list[str] = []

        for i in range(batch_size):
            pil = _tensor_to_pil(image[i])
            r: TagResult = tagger.tag(
                pil,
                threshold=threshold,
                character_threshold=character_threshold,
                uncensored_threshold=uncensored_threshold,
                censored_threshold=censored_threshold,
                genital_threshold=genital_threshold,
                replace_underscore=replace_underscore,
                trailing_comma=trailing_comma,
                exclude_tags=exclude_tags,
            )

            tag_strings.append(r.tag_string)
            flags_uncensored.append(r.label == "uncensored")
            flags_censored.append(r.label == "censored")
            flags_no_explicit.append(r.label == "no_explicit")
            flags_unknown.append(r.label == "unknown")
            ui_tags.append(r.tag_string)
            ui_labels.append(
                f"{r.label}  u={r.uncensored_score:.3f}  "
                f"{r.top_censor_tag}={r.max_censor_score:.3f}  "
                f"{r.top_genital_tag}={r.genital_score:.3f}"
            )

            if pbar is not None:
                pbar.update(1)

        return {
            "ui": {"tags": ui_tags, "labels": ui_labels},
            "result": (
                tag_strings,
                flags_uncensored,
                flags_censored,
                flags_no_explicit,
                flags_unknown,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "OnnxTagger|onnxtagger": OnnxTagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxTagger|onnxtagger": "ONNX Tagger (Censor Detector)",
}
