from src.modules.dynamic.dynamic_module import DynamicModule
from src.modules.signatures.yara_matching import YaraMatcher
from data.configs.config import *
from src.modules.static.static_module import StaticModule


class AISystem:
    def __init__(self, default=False, baseline=False):
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
                model_name="XGB", fetch_pretrained=True, pretrained_path=default_xgb_model
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
                model_name="XGB", fetch_pretrained=True, pretrained_path=baseline_xgb_model
            )
            self.dynamic_module = DynamicModule(
                fetch_pretrained=True,
                model_path=baseline_nebula_model,
                bpe_model_path=baseline_bpe_model,
                vocab_path=baseline_vocab,
            )


    def predict(self, x, separate_scores=False, filter = True):
        
        # scores = [] if separate_scores else None

        white_match = self.white_filter.predict(x)
        if len(white_match) == 1:
            # if separate_scores:
            #     scores.append(("white_list", 0))
            # else:
                return "white_list", 0
        # elif separate_scores:
        #     scores.append(("white_list", None))

        black_match = self.black_filter.predict(x)
        if len(black_match) != 0:
            # if separate_scores:
            #     scores.append(("black_list", 1))
            # else:
                return "black_list", 1
        # elif separate_scores:
        #     scores.append(("black_list", None))

        # Static
        static_score = self.static_module.predict(x)

        # if not filter:
        #     scores.append(("static", static_score[0, 1]))

        if static_score[0, 1] < default_threshold:
            # if separate_scores:
            #     scores.append(("static", 0))
            # else:
                return "static", 0
        elif static_score[0, 1] > (1 - default_threshold):
            # if separate_scores:
            #     scores.append(("static", 1))
            # else:
                return "static", 1
        # else:
        #     if separate_scores:
        #         scores.append(("static", None))

        # Dynamic
        dynamic_score = self.dynamic_module.predict(x)
        # if separate_scores:
        #     scores.append(("dynamic", dynamic_score))
        #     return scores
        return "dynamic", dynamic_score
    
    
    def predict_slifer(self, x):
        white_match = self.white_filter.predict(x)
        if len(white_match) == 1:
            return "white_list", 0
        black_match = self.black_filter.predict(x)
        if len(black_match) != 0:
            return "black_list", 1
        static_score = self.static_module.predict(x)
        if static_score[0, 1] > baseline_xgb_threshold:
            return "static", 1
        dynamic_score = self.dynamic_module.predict(x)
        if dynamic_score == -1:
            return "dynamic", -1
        if dynamic_score > baseline_nebula_threshold:
            return "dynamic", 1
        else:
            return "dynamic", 0



