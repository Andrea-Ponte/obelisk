from src.modules.attacks.ai_system_wrapper import AISystemWrapper

#init model surrogates for the attack on the TMs. The thresholds are set randomly for this example.

tm1_target = AISystemWrapper(
    lgbm_path="path_to_anderson_lgbm_model.txt",
    filter=False,
    threshold=0.80 #set the threshold used to determine if the attack evades the target
)

tm2_target = AISystemWrapper(
    xgb_path="data/models/xgb_with_filters.json", #or xgb_path="data/models/xgb_no_filters.json" if attacking the model in the baseline
    filter=False,
    threshold=0.75

)

tm3_target = AISystemWrapper(
    lgbm_path="path_to_anderson_lgbm_model.json",
    filter=True,
    threshold=0.80
)

tm4_target = AISystemWrapper(
    xgb_path="data/models/xgb_with_filters.json", #or xgb_path="data/models/xgb_no_filters.json" if attacking the model in the baseline
    filter=True,
    threshold=0.70
)

adv_gamma = tm1_target.gamma_section_injection_single(
    malware_sample_path="path_to_malware_sample.exe",
    adv_folder="path_to_adv_folder",
    goodware_folder="path_to_goodware_folder",
    sections=50
)

adv_padding = tm1_target.padding_attack_single(
    malware_sample_path="path_to_malware_sample.exe",
    bytes_to_append=1024,
    adv_folder="path_to_adv_folder"
)
