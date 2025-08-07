from src.modules.attacks.ai_system_wrapper import AISystemWrapper



#init model surrogates for the attack on the TMs.

tm1_surrogate = AISystemWrapper(
    lgbm_path="path_to_anderson_lgbm_model.txt",
    filter=False
)

tm2_surrogate = AISystemWrapper(
    xgb_path="data/models/xgb_with_filters.json", #or xgb_path="data/models/xgb_no_filters.json" if attacking the model in the baseline
    filter=False
)

tm3_surrogate = AISystemWrapper(
    lgbm_path="path_to_anderson_lgbm_model.json",
    filter=True
)

# initialization of TM4
tm4_surrogate = AISystemWrapper(
    xgb_path="data/models/xgb_with_filters.json", #or xgb_path="data/models/xgb_no_filters.json" if attacking the model in the baseline
    filter=True)

adv_gamma = tm1_surrogate.gamma_section_injection_single(
    malware_sample_path="path_to_malware_sample.exe",
    adv_folder="path_to_adv_folder",
    goodware_folder="path_to_goodware_folder",
    sections=50,
    threshold=0.80
)

adv_paddng = tm1_surrogate.padding_attack_single(
    malware_sample_path="path_to_malware_sample.exe",
    bytes_to_append=100,
    adv_folder="path_to_adv_folder"
)
