"""Tests for predictor inference backends."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from predictor.export import export_model
from predictor.features import FeatureVector
from predictor.inference import ONNXPredictor, TorchScriptPredictor, _load_norm_arrays, load_predictor
from predictor.trainer import TrainConfig, Trainer
from predictor.dataset import OutputLengthDataset


@pytest.fixture
def exported_models(tmp_path):
    builder = OutputLengthDataset()
    samples = builder.generate_synthetic(64)
    splits = builder.split(samples)
    cfg = TrainConfig(
        epochs=1,
        batch_size=16,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        use_amp=False,
    )
    result = Trainer(cfg).train(splits)
    models_dir = tmp_path / "models"
    export_model(result.checkpoint_path, models_dir, splits.feature_mean, splits.feature_std)
    return models_dir


@pytest.fixture
def mock_ort(mock_onnx_session):
    mod = MagicMock()
    mod.InferenceSession.return_value = mock_onnx_session
    return mod


@pytest.fixture
def mock_onnx_session():
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="features")]
    session.run.return_value = [np.array([[42.0]], dtype=np.float32)]
    return session


@pytest.mark.unit
def test_onnx_predictor(exported_models, mock_ort, mock_onnx_session):
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        pred = ONNXPredictor(exported_models / "output_length.onnx", exported_models / "metadata.json")
        value, ms = pred.predict(FeatureVector.zeros())
        assert value == 42.0
        assert ms >= 0


@pytest.mark.unit
def test_onnx_predictor_no_metadata(mock_ort):
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        pred = ONNXPredictor("model.onnx", None)
        value, _ = pred.predict(FeatureVector.zeros())
        assert isinstance(value, float)


@pytest.mark.unit
def test_torchscript_predictor(exported_models):
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(item=lambda: 12.0)
    with patch("predictor.inference.torch.jit.load", return_value=mock_model):
        pred = TorchScriptPredictor(
            exported_models / "output_length.ts",
            exported_models / "metadata.json",
        )
        value, ms = pred.predict(FeatureVector.zeros())
        assert value == 12.0
        assert ms >= 0


@pytest.mark.unit
def test_load_predictor_torchscript(exported_models):
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(item=lambda: 8.0)
    with patch("predictor.inference.torch.jit.load", return_value=mock_model):
        ts = load_predictor(exported_models, backend="torchscript")
        assert isinstance(ts, TorchScriptPredictor)


@pytest.mark.unit
def test_load_predictor_backends(exported_models, mock_ort):
    pt = load_predictor(exported_models, backend="pytorch")
    assert pt is not None
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        onnx = load_predictor(exported_models, backend="onnx")
        assert isinstance(onnx, ONNXPredictor)


@pytest.mark.unit
def test_load_norm_arrays(exported_models):
    mean, std = _load_norm_arrays(exported_models / "metadata.json")
    assert mean is not None
    assert std is not None
