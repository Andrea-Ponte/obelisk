from src.modules.dynamic.dynamic_module import DynamicModule
from src.modules.signatures.yara_matching import YaraMatcher
from src.modules.static.static_module import StaticModule
from pathlib import Path

default_nebula_threshold = 0.9999479055404664
default_threshold = 0.03728
baseline_xgb_threshold = 0.964414
baseline_nebula_threshold = 0.9934418201446532

baseline_xgb_model = str(
    Path(__file__).parent.parent.parent.parent / "data/models/xgb_no_filters.json"
)
baseline_vocab = str(
    Path(__file__).parent.parent.parent.parent / "data/models/nebula_ablation_data/baseline/bpe_vocab.json"
)
baseline_bpe_model = str(
    Path(__file__).parent.parent.parent.parent / "data/models/nebula_ablation_data/baseline/bpe.model"
)
baseline_nebula_model = str(
    Path(__file__).parent.parent.parent.parent / "data/models/nebula_ablation_data/baseline/dynamic_model.pt"
)
default_xgb_model = str(
    Path(__file__).parent.parent.parent.parent / "data/models/xgb_with_filters.json"
)
default_vocab = str(
    Path(__file__).parent.parent.parent.parent / "data/models/nebula_ablation_data/delta_11/bpe_vocab.json"
)
default_bpe_model = str(
    Path(__file__).parent.parent.parent.parent / "data/models/nebula_ablation_data/delta_11/bpe.model"
)
default_nebula_model = str(
    Path(__file__).parent.parent.parent.parent / "data/models/nebula_ablation_data/delta_11/dynamic_model.pt"
)
default_whitelist = str(
    Path(__file__).parent.parent.parent.parent / "data/allowlist_rules/windows_files.yar"
)
default_blacklist = str(Path(__file__).parent.parent.parent.parent / "data/blocklist_rules")


class AISystem:
    """Pipeline that combines rule-based, static, and dynamic malware detection."""

    def __init__(
            self,
            baseline=False,
            static_model_path=None,
            dynamic_model_path=None,
            bpe_model_path=None,
            vocab_path=None,
            static_malware_threshold=None,
            static_goodware_threshold=None,
            dynamic_threshold=None,
    ):
        """Initialize the system in baseline mode or with explicit custom model paths."""
        self.white_filter = None
        self.black_filter = None
        self.static_module = None
        self.dynamic_module = None
        self.static_malware_threshold = None
        self.static_goodware_threshold = None
        self.dynamic_threshold = dynamic_threshold
        self.baseline = baseline

        self.init_system(
            baseline=baseline,
            static_model_path=static_model_path,
            dynamic_model_path=dynamic_model_path,
            bpe_model_path=bpe_model_path,
            vocab_path=vocab_path,
            static_malware_threshold=static_malware_threshold,
            static_goodware_threshold=static_goodware_threshold,
            dynamic_threshold=dynamic_threshold,
        )

    def init_system(
            self,
            baseline=False,
            static_model_path=None,
            dynamic_model_path=None,
            bpe_model_path=None,
            vocab_path=None,
            static_malware_threshold=None,
            static_goodware_threshold=None,
            dynamic_threshold=None,
    ):
        """Load filters and models, then configure thresholds for the selected mode."""
        if baseline:
            self.static_goodware_threshold = static_goodware_threshold
            self.static_malware_threshold = (
                baseline_xgb_threshold
                if static_malware_threshold is None
                else static_malware_threshold
            )
            self.dynamic_threshold = (
                baseline_nebula_threshold
                if dynamic_threshold is None
                else dynamic_threshold
            )

            selected_static_model = static_model_path or baseline_xgb_model
            selected_dynamic_model = dynamic_model_path or baseline_nebula_model
            selected_bpe_model = bpe_model_path or baseline_bpe_model
            selected_vocab = vocab_path or baseline_vocab
        else:
            self.static_goodware_threshold = (
                default_threshold if static_goodware_threshold is None else static_goodware_threshold
            )
            self.static_malware_threshold = (
                (1 - default_threshold) if static_malware_threshold is None else static_malware_threshold
            )
            self.dynamic_threshold = dynamic_threshold

            missing_params = []
            if static_model_path is None:
                missing_params.append("static_model_path")
            if dynamic_model_path is None:
                missing_params.append("dynamic_model_path")
            if bpe_model_path is None:
                missing_params.append("bpe_model_path")
            if vocab_path is None:
                missing_params.append("vocab_path")

            if missing_params:
                raise ValueError(
                    "When baseline=False you must pass all model paths in the constructor. Missing: "
                    + ", ".join(missing_params)
                )

            selected_static_model = static_model_path
            selected_dynamic_model = dynamic_model_path
            selected_bpe_model = bpe_model_path
            selected_vocab = vocab_path

        self.white_filter = YaraMatcher(path_to_model=default_whitelist)
        self.black_filter = YaraMatcher(path_to_model=default_blacklist)
        self.static_module = StaticModule(
            model_name="XGB", fetch_pretrained=True, pretrained_path=selected_static_model
        )
        self.dynamic_module = DynamicModule(
            fetch_pretrained=True,
            model_path=selected_dynamic_model,
            bpe_model_path=selected_bpe_model,
            vocab_path=selected_vocab,
        )

    def predict(self, x, separate_scores=False, filter=True):
        """Return the first final decision from rules/static checks, or dynamic output otherwise."""

        white_match = self.white_filter.predict(x)
        if len(white_match) == 1:
            return "white_list", 0

        black_match = self.black_filter.predict(x)
        if len(black_match) != 0:
            return "black_list", 1

        static_score = self.static_module.predict(x)

        if (
                self.static_goodware_threshold is not None
                and static_score[0, 1] < self.static_goodware_threshold
        ):
            return "static", 0
        elif static_score[0, 1] > self.static_malware_threshold:
            return "static", 1

        dynamic_score = self.dynamic_module.predict(x)
        if self.dynamic_threshold is not None:
            if dynamic_score == -1:
                return "dynamic", -1
            return "dynamic", 1 if dynamic_score > self.dynamic_threshold else 0
        return "dynamic", dynamic_score

    def predict_separate(self, x, separate_scores=True):
        """Return intermediate outputs from each stage for analysis or debugging."""

        scores = []

        white_match = self.white_filter.predict(x)
        scores += ["white_list", 1 if len(white_match) == 1 else None]
        if not separate_scores and len(white_match) == 1:
            return "white_list", 0

        black_match = self.black_filter.predict(x)
        scores += ["black_list", 1 if len(black_match) != 0 else None]
        if not separate_scores and len(black_match) != 0:
            return "black_list", 1

        static_score = self.static_module.predict(x)
        static_val = static_score[0, 1]
        scores += ["static", static_val]

        dynamic_score = self.dynamic_module.predict(x)
        scores += ["dynamic", dynamic_score]

        if separate_scores:
            return scores

        return "dynamic", dynamic_score






