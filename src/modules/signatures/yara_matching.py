import os
from typing import Optional

import torch
import yara
from src.utils.module import Module

import pandas as pd
from pathlib import Path


no_fp_rules_path = str(
    Path(__file__).parent.parent.parent.parent / "data/models/rules_with_no_fp.csv"
)

no_fp_rules = (
    pd.read_csv(
        no_fp_rules_path,
        header=None,
    )[0]
    .astype(str)
    .tolist()
)


class YaraMatcher(Module):
    def __init__(
        self,
        path_to_model: Optional[str] = None,
        no_fp=True,
    ):
        # path to model refers to the directory containing the rules
        super().__init__(path_to_model)
        self.matcher = self.load_pretrained_model(self.path_to_model)
        self.no_fp = no_fp

    @staticmethod
    def load_pretrained_model(rules_path: str) -> yara.Rule:
        if rules_path is None:
            raise ValueError("Rules path must be provided and must be a directory.")
        elif not os.path.isdir(rules_path):
            return yara.compile(rules_path)
        else:
            # building rules dictionary
            file_paths = [
                os.path.join(rules_path, file)
                for file in os.listdir(rules_path)
                if os.path.isfile(os.path.join(rules_path, file))
            ]
            paths_dict = {key: key for key in file_paths}
            # compiling rules
            return yara.compile(filepaths=paths_dict)

    def predict(self, x) -> torch.Tensor:
        matches = []
        if isinstance(x, str) or isinstance(x, Path):
            with open(x, "rb") as f:
                x = f.read()
        if isinstance(x, torch.Tensor):
            x = x.data.to(torch.uint8).numpy().tobytes()
        try:
            matches = self.matcher.match(data=x)
            if self.no_fp:
                matches = [rule for rule in matches if str(rule) in no_fp_rules]
            triggered_rules = len(matches)
            # this is useful to convert triggered rules in 1 as prediction result
            if triggered_rules != 0:
                match = 1
            else:
                match = 0
        except Exception as e:
            match = -1
        return matches
