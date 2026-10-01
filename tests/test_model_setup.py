import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tagger_core


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.model = self.directory / tagger_core.MODEL_BASENAME
        self.csv = self.directory / tagger_core.CSV_BASENAME

    @staticmethod
    def download(url, dest, label=None):
        dest.write_bytes(b"downloaded")

    def test_downloads_fp16_without_conversion_dependencies(self):
        with mock.patch.object(tagger_core, "_download", side_effect=self.download) as download:
            with mock.patch.dict(sys.modules, {"onnx": None, "onnxconverter_common": None}):
                result = tagger_core._ensure_fp16_model(self.directory)
        self.assertEqual(result, (self.model, self.csv))
        self.assertEqual([call.args for call in download.call_args_list], [
            (tagger_core.HF_CSV_URL, self.csv),
            (tagger_core.HF_FP16_URL, self.model),
        ])
        self.assertTrue(self.model.exists())
        self.assertFalse((self.directory / "wd-eva02-large-tagger-v3.onnx").exists())

    def test_reuses_existing_model_and_csv(self):
        self.model.write_bytes(b"existing model")
        self.csv.write_bytes(b"existing csv")
        with mock.patch.object(tagger_core, "_download") as download:
            result = tagger_core._ensure_fp16_model(self.directory)
        download.assert_not_called()
        self.assertEqual(result, (self.model, self.csv))
        self.assertEqual(self.model.read_bytes(), b"existing model")

    def test_only_downloads_missing_csv(self):
        self.model.write_bytes(b"existing model")
        with mock.patch.object(tagger_core, "_download", side_effect=self.download) as download:
            tagger_core._ensure_fp16_model(self.directory)
        download.assert_called_once_with(tagger_core.HF_CSV_URL, self.csv, label="selected_tags.csv")
        self.assertEqual(self.model.read_bytes(), b"existing model")

    def test_download_failure_is_not_hidden_by_conversion(self):
        self.csv.write_bytes(b"existing csv")
        with mock.patch.object(tagger_core, "_download", side_effect=OSError("download failed")):
            with self.assertRaisesRegex(OSError, "download failed"):
                tagger_core._ensure_fp16_model(self.directory)
        self.assertFalse(self.model.exists())

    def test_retains_fp32_download_and_conversion(self):
        self.csv.write_bytes(b"existing csv")
        source = self.directory / "wd-eva02-large-tagger-v3.onnx"
        with mock.patch.object(tagger_core, "HF_FP16_URL", None):
            with mock.patch.object(tagger_core, "_download", side_effect=self.download) as download:
                with mock.patch("onnx.load") as load, mock.patch("onnx.save") as save:
                    with mock.patch("onnxconverter_common.float16.convert_float_to_float16") as convert:
                        save.side_effect = lambda model, path: Path(path).write_bytes(b"converted")
                        result = tagger_core._ensure_fp16_model(self.directory)
        download.assert_called_once_with(
            tagger_core.HF_ONNX_URL, source, label="model.onnx (FP32, 約 1.2 GB)"
        )
        load.assert_called_once_with(str(source))
        convert.assert_called_once_with(
            load.return_value,
            op_block_list=["LayerNormalization", "Softmax", "Sigmoid", "ReduceMean", "Div", "Conv"],
            keep_io_types=True,
            disable_shape_infer=True,
        )
        self.assertEqual(result, (self.model, self.csv))
        self.assertEqual(self.model.read_bytes(), b"converted")
        self.assertFalse(source.exists())
        self.assertFalse(self.model.with_suffix(".onnx.part").exists())


if __name__ == "__main__":
    unittest.main()
