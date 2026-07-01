"""Feature vector edge-case tests."""

import pytest
from predictor.features import FEATURE_NAMES, NUM_FEATURES, FeatureVector


@pytest.mark.unit
def test_feature_vector_wrong_length():
    with pytest.raises(ValueError):
        FeatureVector(values=(1.0, 2.0))


@pytest.mark.unit
def test_feature_vector_as_dict_roundtrip():
    data = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    fv = FeatureVector.from_dict(data)
    assert len(fv.as_list()) == NUM_FEATURES
    assert fv.as_dict()["prompt_token_count"] == 0.0
