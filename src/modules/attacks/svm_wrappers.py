from abc import abstractmethod

from secml.array import CArray
from secml.ml.classifiers import CClassifier

import numpy as np
from ember import PEFeatureExtractor
from secml.array import CArray
from secml.ml.classifiers import CClassifier

from secml.ml.classifiers.sklearn.c_classifier_sklearn import CClassifierSkLearn

import joblib

from sklearn.preprocessing import MinMaxScaler


class CWrapperPhi:
    """
    Abstract class that encapsulates a model for being used in a black-box way.
    """

    def __init__(self, model: CClassifier):
        self.classifier = model

    @abstractmethod
    def extract_features(self, x: CArray):
        raise NotImplementedError(
            "This method is abstract, you should implement it somewhere else!"
        )

    def predict(self, x: CArray, return_decision_function: bool = True):
        x = x.atleast_2d()
        # feature_vectors = []
        # for i in range(x.shape[0]):
        # 	x_i = x[i, :]
        # 	padding_position = x_i.find(x_i == 256)
        # 	if padding_position:
        # 		x_i = x_i[0, :padding_position[0]]
        # 	feature_vectors.append(self.extract_features(x_i))
        # feature_vectors = CArray(feature_vectors)
        feature_vectors = self.extract_features(x)
        if (feature_vectors == CArray.zeros((x.shape[0], 2381))).all():
            yara_score = np.array([[0.0, 7.0]])
            yara_score = (1, CArray(yara_score))
            return yara_score
        return self.classifier.predict(
            feature_vectors, return_decision_function=return_decision_function
        )


class CEmberSVMWrapperPhi(CWrapperPhi):
    def __init__(
        self,
        model: CClassifierSkLearn,
        scaler: MinMaxScaler = None,
        filter=False,
        white_filter=None,
        black_filter=None,
    ):
        if not isinstance(model, CClassifierSkLearn):
            raise ValueError(f"Input model is {type(model)} and not CClassifier")
        super().__init__(model)
        if scaler is not None:
            self.scaler = scaler
        self.filter = filter
        if filter:
            self.white_filter = white_filter
            self.black_filter = black_filter

    def extract_features(self, x):
        extractor = PEFeatureExtractor(2, print_feature_warning=False)
        x = x.atleast_2d()
        size = x.shape[0]
        features = []
        x_i = x[0, :]
        x_bytes = bytes(x_i.astype(np.uint8).tolist()[0])
        if self.filter:
            yara_white = self.white_filter.predict(x_bytes)
            yara_black = self.black_filter.predict(x_bytes)
            if len(yara_black) != 0:
                return CArray.zeros((x.shape[0], 2381))

        for i in range(size):
            x_i = x[i, :]
            length = x_i.find(x_i == 256)
            if length:
                x_i = x_i[0, : length[0]]
            x_bytes = bytes(x_i.astype(np.uint8).tolist()[0])
            vector = np.array(extractor.feature_vector(x_bytes), dtype=np.float32)
            vector = self.scaler.transform(vector.reshape(1, -1)).reshape(-1)
            features.append(vector)
        features = CArray(features)
        return features
