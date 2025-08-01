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
    def __init__(self, default=True, baseline=False):
        self.white_filter = None
        self.black_filter = None
        self.static_module = None
        self.dynamic_module = None

        if default:
            self.init_system(default=default)
        elif baseline:
            self.init_system(baseline=baseline)
        else:
            raise ValueError(
                "You must specify either default=True or baseline=True to initialize the AISystem."
            )

    def init_system(self, default=False, baseline=False):
        if default:
            self.white_filter = YaraMatcher(path_to_model=default_whitelist)
            self.black_filter = YaraMatcher(path_to_model=default_blacklist)
            self.static_module = StaticModule(
                model_name="XGB",
                fetch_pretrained=True,
                pretrained_path=default_xgb_model,
            )
            self.dynamic_module = DynamicModule(
                fetch_pretrained=True,
                model_path=default_nebula_model,
                bpe_model_path=default_bpe_model,
                vocab_path=default_vocab,
            )
        if baseline:
            self.white_filter = YaraMatcher(path_to_model=default_whitelist)
            self.black_filter = YaraMatcher(path_to_model=default_blacklist)
            self.static_module = StaticModule(
                model_name="XGB",
                fetch_pretrained=True,
                pretrained_path=baseline_xgb_model,
            )
            self.dynamic_module = DynamicModule(
                fetch_pretrained=True,
                model_path=baseline_nebula_model,
                bpe_model_path=baseline_bpe_model,
                vocab_path=baseline_vocab,
            )

    # prediction method of OBELISK
    def predict(self, x):
        white_match = self.white_filter.predict(x)
        if len(white_match) == 1:
            return "white_list", 0

        black_match = self.black_filter.predict(x)
        if len(black_match) != 0:
            return "black_list", 1

        # Static
        static_score = self.static_module.predict(x)

        if static_score[0, 1] <= default_threshold:
            return "static", 0
        elif static_score[0, 1] >= (1 - default_threshold):
            return "static", 1

        # Dynamic
        dynamic_score = self.dynamic_module.predict(x)

        if dynamic_score == -1:
            return "dynamic", -1
        if dynamic_score >= default_nebula_threshold:
            return "dynamic", 1
        else:
            return "dynamic", 0

    # prediction method of SLIFER (Ponte et al. 2025), where malware are halted as soon as
    # a module detects it, while goodware are processed by all modules
    def predict_slifer(self, x):
        white_match = self.white_filter.predict(x)
        if len(white_match) == 1:
            return "white_list", 0
        black_match = self.black_filter.predict(x)
        if len(black_match) != 0:
            return "black_list", 1
        static_score = self.static_module.predict(x)
        if static_score[0, 1] >= baseline_xgb_threshold:
            return "static", 1
        dynamic_score = self.dynamic_module.predict(x)
        if dynamic_score == -1:
            return "dynamic", -1
        if dynamic_score >= baseline_nebula_threshold:
            return "dynamic", 1
        else:
            return "dynamic", 0
