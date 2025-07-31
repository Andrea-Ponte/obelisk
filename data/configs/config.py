from pathlib import Path
import pandas as pd

default_blacklist = str(Path(__file__).parent.parent / "blacklist_rules_v2")

default_whitelist = str(Path(__file__).parent.parent / "whitelist_rules_v2")

no_fp_rules = (
    pd.read_csv(
        # "/Users/bridge/PhD/Code/obelisk/data/results/V2/yara/rules_with_no_fp.csv",
        "/data/aponte/repos/obelisk/data/results/V2/yara/rules_with_no_fp.csv",
        header=None,
    )[0]
    .astype(str)
    .tolist()
)

default_whitelist = str(
    Path(__file__).parent.parent / "whitelist_rules/windows_files.yar"
)
default_xgb_model = str(
    Path(__file__).parent.parent / "models/V2/xgb/xgb_with_filters.json"
)
default_vocab = str(
    Path(__file__).parent.parent / "nebula_ablation_data/delta_11/bpe_vocab.json"
)
default_bpe_model = str(
    Path(__file__).parent.parent / "nebula_ablation_data/delta_11/bpe.model"
)
default_nebula_model = str(
    Path(__file__).parent.parent / "nebula_ablation_data/delta_11/dynamic_model.pt"
)
default_threshold = 0.03728

default_nebula_threshold = 0.9999479055404664

baseline_xgb_model = str(Path(__file__).parent.parent / "models/V2/xgb/xgb_no_filters.json")

baseline_vocab = str( Path(__file__).parent.parent / "models/V2/nebula/baseline/bpe_vocab.json")

baseline_bpe_model = str(Path(__file__).parent.parent / "models/V2/nebula/baseline/bpe.model")

baseline_nebula_model = str(Path(__file__).parent.parent / "models/V2/nebula/baseline/dynamic_model.pt")

baseline_xgb_threshold = 0.964414

baseline_nebula_threshold = 0.9934418201446532

threshold_rules_no_filters = 0.963746190071106