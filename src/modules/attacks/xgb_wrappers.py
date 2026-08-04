
from abc import abstractmethod
from pathlib import Path

import numpy as np
from ember import PEFeatureExtractor
from lightgbm import Booster
from secml.array import CArray
from secml.ml.classifiers import CClassifier
from xgboost import XGBClassifier

from src.modules.signatures.yara_matching import YaraMatcher

default_whitelist = str(
    Path(__file__).parent.parent / "allowlist_rules/windows_files.yar"
)
default_blacklist = str(Path(__file__).parent.parent / "blacklist_rules_v2")


def _strip_padding(x_row: CArray) -> CArray:
    """Truncate a (1, n) row at the first padding sentinel, if present."""
    padding_positions = x_row.find(x_row == 256)
    if padding_positions:
        return x_row[0, : padding_positions[0]]
    return x_row


def _row_to_bytes(x_row: CArray) -> bytes:
    """Convert a single (1, n) uint8 CArray row into raw bytes."""
    return bytes(x_row.astype(np.uint8).tolist()[0])


class CWrapperPhi:
    """Abstract base that lets black-box attacks operate on raw byte arrays:
    subclasses implement extract_features() to turn raw bytes into the
    wrapped classifier's real feature space, and predict() feeds that
    through to the classifier."""

    def __init__(self, model: CClassifier):
        self.classifier = model

    @abstractmethod
    def extract_features(self, x: CArray):
        raise NotImplementedError(
            "This method is abstract, you should implement it somewhere else!"
        )

    def predict(self, x: CArray, return_decision_function: bool = True):
        x = x.atleast_2d()
        feature_vectors = self.extract_features(x)
        return self.classifier.predict(
            feature_vectors, return_decision_function=return_decision_function
        )


class CClassifierXGBoost(CClassifier):
    """secml classifier wrapping either an XGBoost or a LightGBM static model,
    with an optional YARA-blacklist short-circuit ahead of the model."""

    def __init__(
        self,
        xgb_path: str = None,
        filter: bool = False,
        lgbm_path: str = None,
    ):
        super().__init__()
        if lgbm_path is not None:
            self._model = Booster(model_file=lgbm_path)
            self.model_name = "lgbm"
        else:
            self._model = self._load_xgb_model(xgb_path)
            self.model_name = "xgb"

        self.filter = filter
        if self.filter:
            # Built once here instead of on every extract_features() call.
            self._black_filter = YaraMatcher(path_to_model=default_blacklist)

    def extract_features(self, x: CArray) -> CArray:
        """Extract an EMBER static feature vector per sample (row) in x.

        If self.filter is enabled, the batch is first checked against the
        YARA blacklist using only its first sample: a match short-circuits
        the whole batch and returns an all-zero feature matrix, which
        predict() below recognizes as "force malware verdict". This assumes
        single-sample batches, which holds during the black-box attack loop.
        """
        extractor = PEFeatureExtractor(2, print_feature_warning=False)
        x = x.atleast_2d()
        size = x.shape[0]

        if self.filter:
            first_sample_bytes = _row_to_bytes(x[0, :])
            if len(self._black_filter.predict(first_sample_bytes)) != 0:
                return CArray.zeros((size, 2381))

        features = []
        for i in range(size):
            x_i = _strip_padding(x[i, :])
            x_bytes = _row_to_bytes(x_i)
            features.append(
                np.array(extractor.feature_vector(x_bytes), dtype=np.float32)
            )

        return CArray(features)

    def _backward(self, w):
        pass

    def _fit(self, x, y):
        raise NotImplementedError("Fit is not implemented.")

    def _load_xgb_model(self, xgb_path: str) -> XGBClassifier:
        model = XGBClassifier()
        model.load_model(xgb_path)
        return model

    def _forward(self, x):
        # Not used directly: predict() below overrides the public API and
        # bypasses secml's internal decision_function dispatch. Kept in case
        # secml_malware's internals still call it through decision_function().
        x = x.atleast_2d()
        scores = self._model.predict_proba(x.tondarray())
        confidence = [[1 - c, c] for c in scores]
        return CArray(confidence)

    def predict(self, x: CArray, return_decision_function: bool = False):
        """Return (flag, scores):
          - flag == 1: the batch was force-classified as malware by the YARA
            blacklist filter in extract_features() (detected via the
            all-zero feature-vector sentinel below).
          - flag == 0: scores come from the underlying xgb/lgbm model.

        return_decision_function is accepted for interface compatibility
        with CClassifier.predict but isn't used here, since this override
        doesn't go through secml's internal decision_function dispatch.
        """
        if (x == CArray.zeros((x.shape[0], EMBER_FEATURE_VECTOR_SIZE))).all():
            return 1, CArray(np.array([[0.0, 1.0]]))

        scores = (
            self._model.predict_proba(x.tondarray())
            if self.model_name == "xgb"
            else self._model.predict(x.tondarray())
        )
        if self.model_name == "lgbm":
            scores = [[1 - scores[0], scores[0]]]

        return 0, CArray(scores)


class CXGBWrapperPhi(CWrapperPhi):
    """CWrapperPhi specialization for CClassifierXGBoost models."""

    def __init__(self, model: CClassifierXGBoost):
        if not isinstance(model, CClassifierXGBoost):
            raise ValueError(f"Input model is {type(model)} and not CClassifierXGBoost")
        super().__init__(model)

    def extract_features(self, x: CArray) -> CArray:
        x = x.atleast_2d()
        clf: CClassifierXGBoost = self.classifier
        feature_vectors = CArray.zeros((x.shape[0], EMBER_FEATURE_VECTOR_SIZE))
        for i in range(x.shape[0]):
            x_i = _strip_padding(x[i, :])
            feature_vectors[i, :] = clf.extract_features(x_i)
        return feature_vectors