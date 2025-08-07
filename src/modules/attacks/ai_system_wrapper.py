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
from src.modules.attacks.xgb_wrappers import CClassifierXGBoost, CXGBWrapperPhi


class AISystemWrapper:
    def __init__(
        self,
        xgb_path=None,
        filter=False,
        lgbm_path=None,
        threshold=None,
    ):
        model = CClassifierXGBoost(
            lgbm_path=lgbm_path,
            xgb_path=xgb_path,
            filter=filter,
        )
        model = CXGBWrapperPhi(model)
        self.ai_system = model
        self.threshold = threshold

    def gamma_section_injection_single(
        self,
        malware_sample_path: str,
        adv_folder,
        goodware_folder: str = None,
        sections: int = 50,
    ):
        section_population, what_from_who = (
            CGammaSectionsEvasionProblem.create_section_population_from_folder(
                goodware_folder,
                how_many=sections,
                sections_to_extract=[".rdata"],
                to_ignore=[],
            )
        )
        attack = CGammaSectionsEvasionProblem(
            section_population,
            self.ai_system,
            population_size=10,
            penalty_regularizer=1e-7,
            iterations=sections,
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
        self, malware_sample_path: str, adv_folder, bytes_to_append, threshold=None
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
        if adv_score < threshold:
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


    # Multiprocessing attack starter
    def mp_attack_starter(
        self,
        malware_samples: list,
        adv_folder: str,
        n_jobs,
        which_attack,
        goodware_folder: str = None,
        bytes_to_append=None,
    ):
        malware_chunks = [malware_samples[i::n_jobs] for i in range(n_jobs)]
        print(
            f"Splitting {len(malware_samples)} samples into {n_jobs} chunks for multiprocessing."
        )

        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.starmap(
                self.mp_multiple_attack,
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
    def mp_multiple_attack(
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
                cls.gamma_section_injection_single(sample, adv_folder, goodware_folder)
