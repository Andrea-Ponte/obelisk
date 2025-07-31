import multiprocessing
import os
from abc import abstractmethod

from secml.array import CArray
from secml.ml.classifiers import CClassifier

import numpy as np
from ember import PEFeatureExtractor
from secml.array import CArray
from secml.ml.classifiers import CClassifier

from secml.ml.classifiers.sklearn.c_classifier_sklearn import CClassifierSkLearn

import joblib
from secml_malware.attack.blackbox.c_black_box_padding_evasion import (
    CBlackBoxPaddingEvasionProblem,
)
from secml_malware.attack.blackbox.c_gamma_sections_evasion import (
    CGammaSectionsEvasionProblem,
)
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from src.modules.signatures.yara_matching import YaraMatcher
from lightgbm import Booster


# white_filter = YaraMatcher(path_to_model='/data/aponte/repos/obelisk/data/whitelist_rules/windows_files.yar')
# black_filter = YaraMatcher(path_to_model='/data/aponte/repos/obelisk/data/blacklist_rules')


default_win_folder = "/data/aponte/repos/obelisk/data/win_exe/pes/win11/syswow64"
default_choco_folder = "/data/mkozak/chocolatey-10000/chocolatey-selected-10000-EXE"


class CWrapperPhi:
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
        return self.classifier.predict(
            feature_vectors, return_decision_function=return_decision_function
        )


import numpy as np
from ember import PEFeatureExtractor
from secml.array import CArray
from secml.ml.classifiers import CClassifier


class CClassifierXGBoost(CClassifier):
    def __init__(
        self,
        xgb_path: str = None,
        filter=False,
        white_filter=None,
        black_filter=None,
        lgbm_path=None,
    ):
        super(CClassifierXGBoost, self).__init__()
        if lgbm_path is not None:
            self._model = Booster(model_file=lgbm_path)
            self.model_name = "lgbm"
        else:
            self._model = self._load_tree(xgb_path)
            self.model_name = "xgb"
        self.filter = filter
        # if filter:
        # self.white_filter = white_filter
        # self.black_filter = black_filter
        # self.white_filter = YaraMatcher(
        #     path_to_model="/data/aponte/repos/obelisk/data/whitelist_rules/windows_files.yar"
        #      )
        # self.black_filter = YaraMatcher(
        #     path_to_model="/data/aponte/repos/obelisk/data/blacklist_rules_v2"
        # )

    def extract_features(self, x: CArray) -> CArray:
        extractor = PEFeatureExtractor(2, print_feature_warning=False)
        x = x.atleast_2d()
        size = x.shape[0]
        features = []
        x_i = x[0, :]
        x_bytes = bytes(x_i.astype(np.uint8).tolist()[0])
        if self.filter:
            white_filter = YaraMatcher(
                path_to_model="/data/aponte/repos/obelisk/data/whitelist_rules/windows_files.yar"
            )
            black_filter = YaraMatcher(
                path_to_model="/data/aponte/repos/obelisk/data/blacklist_rules_v2"
            )
            # negli attacchi non può succedere
            yara_white = white_filter.predict(x_bytes)
            yara_black = black_filter.predict(x_bytes)
            if len(yara_black) != 0:
                return CArray.zeros((x.shape[0], 2381))
        for i in range(size):
            x_i = x[i, :]
            length = x_i.find(x_i == 256)
            if length:
                x_i = x_i[0, : length[0]]
            x_bytes = bytes(x_i.astype(np.uint8).tolist()[0])
            features.append(
                np.array(extractor.feature_vector(x_bytes), dtype=np.float32)
            )
        features = CArray(features)
        return features

    def _backward(self, w):
        pass

    def _fit(self, x, y):
        raise NotImplementedError("Fit is not implemented.")

    def _load_tree(self, tree_path):
        model = XGBClassifier()
        model.load_model(tree_path)
        return model

    def _forward(self, x):
        x = x.atleast_2d()
        scores = self._xgboost_model.predict_proba(x.tondarray())
        confidence = [[1 - c, c] for c in scores]
        confidence = CArray(confidence)
        return confidence

    def predict(self, x, return_decision_function=False):
        if (x == CArray.zeros((x.shape[0], 2381))).all():  # Hobrigado deus
            yara_score = np.array([[0.0, 1.0]])
            yara_score = (1, CArray(yara_score))
            return yara_score

        scores = (
            self._model.predict_proba(x.tondarray())
            if self.model_name == "xgb"
            else self._model.predict(x.tondarray())
        )
        # print(f"Scores shape: {scores.shape}, type: {type(scores)}")

        if self.model_name == "lgbm":
            scores = [[1 - scores[0], scores[0]]]

        # # Checking if the score is higher than ember model threshold
        # labels = (scores > 0.82).astype(int)

        # label = labels.argmax(axis=1).ravel()

        # Ensure scores is a CArray of shape (1, 2) for compatibility with confidence[0, 1].item()
        # if isinstance(scores, np.ndarray):
        #     scores = CArray(scores)
        # if scores.ndim == 1:
        #     # If scores is (1,), convert to (1, 2) with [1-c, c]
        #     c = scores[0]
        #     scores = [[1 - c, c]]
        return (0, CArray(scores))


class CXGBWrapperPhi(CWrapperPhi):
    def __init__(self, model: CClassifierXGBoost):
        if not isinstance(model, CClassifierXGBoost):
            raise ValueError(f"Input model is {type(model)} and not CClassifierEmber")
        super().__init__(model)

    def extract_features(self, x):
        x = x.atleast_2d()
        clf: CClassifierXGBoost = self.classifier
        feature_vectors = CArray.zeros((x.shape[0], 2381))
        for i in range(x.shape[0]):
            x_i = x[i, :]
            padding_positions = x_i.find(x_i == 256)
            if padding_positions:
                feature_vectors[i, :] = clf.extract_features(
                    x_i[0, : padding_positions[0]]
                )
            else:
                feature_vectors[i, :] = clf.extract_features(x_i)
        return feature_vectors


class AISystemWrapper:
    def __init__(
        self,
        xgb_path=None,
        white_rules=None,
        black_rules=None,
        filter=False,
        lgbm_path=None,
    ):
        # white_filter = YaraMatcher(
        #     path_to_model="/data/aponte/repos/obelisk/data/whitelist_rules/windows_files.yar"
        # )
        # black_filter = YaraMatcher(
        #     path_to_model="/data/aponte/repos/obelisk/data/blacklist_rules_v2"
        # )
        model = CClassifierXGBoost(
            # lgbm_path=lgbm_path,
            xgb_path=xgb_path,
            filter=filter,
            # white_filter=white_filter,
            # black_filter=black_filter,
        )
        model = CXGBWrapperPhi(model)
        self.ai_system = model
        # self.threshold = 0.9982742  # Threshold for FPR ~ 1e-3: 0.99795294 (FPR=0.0009647608408652382)
        # self.threhsold = 0.97363937  # Threshold for FPR ~ 1e-2: 0.97363937 (FPR=0.009901492840459023)
        # self.threshold = 0.964414 # baseline_xgb_threshold
        self.threshold = 0.963746190071106  # baseline with no filters

    def gamma_section_injection(
        self,
        malware_sample_path: str,
        adv_folder,
        goodware_folder: str = default_win_folder,
    ):
        section_population, what_from_who = (
            CGammaSectionsEvasionProblem.create_section_population_from_folder(
                goodware_folder,
                how_many=50,
                sections_to_extract=[".rdata"],
                # to_ignore= []
                to_ignore=[
                    "c7c9e072161fe239e3406e5d45b85eef9b1e0795a4f8219032fa14ec765accfe"
                ],
            )
        )
        attack = CGammaSectionsEvasionProblem(
            section_population,
            self.ai_system,
            population_size=10,
            penalty_regularizer=1e-7,
            iterations=50,
            threshold=0,
        )
        engine = CGeneticAlgorithm(attack)
        with open(malware_sample_path, "rb") as f:
            malware_sample = f.read()
            malware_sample = CArray(np.frombuffer(malware_sample, dtype=np.uint8))
            malware_sample = malware_sample[0, :]
        _, adv_score, adv_ds, _ = engine.run(malware_sample, CArray([1]))
        adv_score = adv_score[-1]
        if adv_score < self.threshold:
            print(f"Success! Score: {adv_score}")
            adv_example = adv_ds.X[0, :]
            malware_hash = malware_sample_path.split("/")[-1]
            print(f"Saving file {malware_hash}_adv")
            engine.write_adv_to_file(adv_example, adv_folder + malware_hash + "_adv")
        else:
            print(f"Failed! Score: {adv_score}")

    def padding_attack_single(
        self, malware_sample_path: str, adv_folder, bytes_to_append
    ):
        attack = CBlackBoxPaddingEvasionProblem(
            self.ai_system,
            population_size=10,
            how_many_padding_bytes=bytes_to_append,
            iterations=50,
        )
        engine = CGeneticAlgorithm(attack)
        with open(malware_sample_path, "rb") as f:
            malware_sample = f.read()
            malware_sample = CArray(
                np.frombuffer(malware_sample, dtype=np.uint8)
            ).atleast_2d()

        _, adv_score, adv_ds, _ = engine.run(malware_sample, CArray([1]))
        adv_score = adv_score[-1]
        if adv_score < self.threshold:
            print(f"Success! Score: {adv_score}")
            adv_example = adv_ds.X[0, :]
            malware_hash = malware_sample_path.split("/")[-1]
            print(f"Saving file {malware_hash}_adv")
            engine.write_adv_to_file(
                adv_example,
                os.path.join(adv_folder, str(bytes_to_append), malware_hash + "_adv"),
            )
        else:
            print(f"Failed! Score: {adv_score}")

    def mp_attack_starter(
        self,
        malware_samples: list,
        adv_folder: str,
        n_jobs,
        which_attack,
        goodware_folder: str = default_win_folder,
        bytes_to_append=None,
    ):
        malware_chunks = [malware_samples[i::n_jobs] for i in range(n_jobs)]
        print(
            f"Splitting {len(malware_samples)} samples into {n_jobs} chunks for multiprocessing."
        )
        # if not os.path.exists(adv_folder):
        #     os.makedirs(adv_folder)
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.starmap(
                self.mp_multiple_transfer_attack,
                [
                    (
                        chunk,
                        adv_folder,
                        which_attack,
                        goodware_folder,
                        self,
                        bytes_to_append,
                    )
                    for chunk in malware_chunks
                ],
            )

    @staticmethod
    def mp_multiple_transfer_attack(
        malware_samples: list,
        adv_folder: str,
        which_attack: str,
        goodware_folder: str,
        cls,
        bytes_to_append=None,
    ):
        for i, sample in enumerate(malware_samples):
            print(f"Processing {i} of {len(malware_samples)}")
            if which_attack == "padding":
                cls.padding_attack_single(sample, adv_folder, bytes_to_append)
            elif which_attack == "gamma":
                cls.gamma_section_injection(sample, adv_folder, goodware_folder)
