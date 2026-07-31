import joblib
import numpy as np
import torch
from ember import PEFeatureExtractor
from lightgbm import Booster
from nebula.models import ember
from secml.array import CArray
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.utils.module import Module
import xgboost as xgb
from pathlib import Path


class StaticModule(Module):
    def __init__(
        self,
        model_name: str,
        hparams: dict = None,
        fetch_pretrained: bool = False,
        pretrained_path: str = None,
    ):
        super().__init__()
        self.model = None
        self.model_name = model_name
        self.scaler = None

        if fetch_pretrained:
            if pretrained_path is None:
                raise ValueError(
                    "pretrained_path must be specified when fetch_pretrained is True"
                )
            self.load_pretrained_model(pretrained_path)
        else:
            if hparams is None:
                raise ValueError(
                    "hparams must be specified when fetch_pretrained is False"
                )
            self.build_model(hparams)

    def build_model(self, hparams):
        if self.model_name == "XGB":
            self.model = XGBClassifier(
                n_estimators=hparams["n_estimators"],
                learning_rate=hparams["learning_rate"],
                objective=hparams["objective"],
                n_jobs=hparams["n_jobs"],
                booster=hparams["booster"],
                colsample_bytree=hparams["colsample_bytree"],
            )
        else:
            raise ValueError("Only XGB is supported for training")

    def load_pretrained_model(self, pretrained_path):
        # option to use Anderson's Ember model
        if self.model_name == "LGBM":
            lgbm = Booster(model_file=pretrained_path)
            self.model = lgbm
        if self.model_name == "XGB":
            xgb = XGBClassifier()
            xgb.load_model(pretrained_path)
            self.model = xgb

        return None

    def train_module(self, X, y):
        if self.model_name == "XGB":
            self.model.fit(X, y)

    def predict(self, x, extract_features=True):
        if self.model_name == "LGBM":
            if extract_features:
                x = self.extract_features(x)
            return self.model.predict(x)
        if self.model_name == "XGB":
            if extract_features:
                x = self.extract_features(x)
            score = self.model.predict_proba(x)
            return score
        return -1

    def save_model(self, path):
        if self.model_name == "XGB":
            if path.endswith(".json"):
                self.model.save_model(path)
        return None

    @staticmethod
    def extract_features(x):
        extractor = PEFeatureExtractor(2, print_feature_warning=False)
        if isinstance(x, str) or isinstance(x, Path):
            with open(x, "rb") as f:
                x = f.read()
                x = np.frombuffer(x, dtype=np.uint8)
        if 256 in x:
            x = x[:x.index(256)]
        x_b = bytes(x)
        features = np.array(extractor.feature_vector(x_b)).reshape(1, -1)
        return features
