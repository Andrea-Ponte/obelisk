import multiprocessing
import os

import numpy as np
from secml.array import CArray
from secml.ml import CClassifier
from secml_malware.attack.blackbox.c_black_box_padding_evasion import (
    CBlackBoxPaddingEvasionProblem,
)
from secml_malware.attack.blackbox.c_gamma_sections_evasion import (
    CGammaSectionsEvasionProblem,
)
from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi
import secml_malware.models.c_classifier_ember as ember
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from secml_malware.attack.blackbox.ga.c_nevergrad_ga import CNevergradGeneticAlgorithm


default_win_folder = "/data/aponte/repos/obelisk/data/win_exe/pes/win11/syswow64"
default_choco_folder = "/data/mkozak/chocolatey-10000/chocolatey-selected-10000-EXE"
default_transfer_padding_adv_folder = (
    "/data/aponte/repos/obelisk/data/adv_exe/V2/transfer/gamma1/choco_sections/"
)


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class OpenGbdt:
    def __init__(self, model_path: str):
        if model_path is None:
            raise ValueError("model_path cannot be None")
        self.classifier = CEmberWrapperPhi(ember.CClassifierEmber(model_path))
        self.threshold = 0.82

    def padding_attack_single(
        self, malware_sample_path: str, bytes_to_append, adv_folder
    ):
        attack = CBlackBoxPaddingEvasionProblem(
            self.classifier,
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
                to_ignore=[
                    "c7c9e072161fe239e3406e5d45b85eef9b1e0795a4f8219032fa14ec765accfe"
                ],
            )
        )
        attack = CGammaSectionsEvasionProblem(
            section_population,
            self.classifier,
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

    def multiple_transfer_attack(
        self,
        malware_samples: list,
        adv_folder: str = default_transfer_padding_adv_folder,
        bytes_to_append=1536,
        which_attack: str = "gamma",
        goodware_folder: str = default_choco_folder,
    ):
        for i, sample in enumerate(malware_samples):
            print(f"Processing sample {i + 1}/{len(malware_samples)}")
            if not os.path.exists(sample):
                print(f"Sample {sample} does not exist. Skipping.")
                continue
            if which_attack == "padding":
                self.padding_attack_single(sample, bytes_to_append, adv_folder)
            if which_attack == "gamma":
                self.gamma_section_injection(sample, adv_folder, goodware_folder)
            if which_attack == "both":
                self.padding_attack_single(sample, bytes_to_append, adv_folder)
                self.gamma_section_injection(sample, adv_folder, goodware_folder)

    def mp_transfer_attack(
        self,
        malware_samples: list,
        adv_folder: str,
        bytes_to_append,
        n_jobs,
        which_attack,
        goodware_folder: str = default_win_folder,
    ):
        malware_chunks = [malware_samples[i::n_jobs] for i in range(n_jobs)]
        print(
            f"Splitting {len(malware_samples)} samples into {n_jobs} chunks for multiprocessing."
        )
        # if not os.path.exists(os.path.join(adv_folder, str(bytes_to_append))):
        #     os.makedirs(adv_folder)
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.map(self.multiple_transfer_attack, malware_chunks)

    def test_attacks(self, folder, results_path):
        evaded = 0
        for example in os.listdir(folder):
            with open(os.path.join(folder, example), "rb") as f:
                malware_sample = f.read()
                malware_sample = CArray(
                    np.frombuffer(malware_sample, dtype=np.uint8)
                ).atleast_2d()
            _, confidence = self.classifier.predict(
                malware_sample, return_decision_function=True
            )
            if confidence[0, 1].item() < self.threshold:
                evaded += 1
                with open(results_path, "a") as f:
                    f.write(f"{example}, {confidence[0, 1].item()}\n")
        print(
            f"Evaded {evaded} out of {len(os.listdir(folder))} samples with threshold {self.threshold}."
        )
        with open(results_path, "a") as f:
            f.write(
                f"Evaded {evaded} out of {len(os.listdir(folder))} samples with threshold {self.threshold}.\n"
            )
