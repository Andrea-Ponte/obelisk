# OBELISK

Repository for sharing code and assets used in the paper "_OBELISK: Understanding How Malware Detection AI Systems Succeed and Fail_".

# Modules
Here we describe the released assets.

- ### YARA Signatures
    The pool of YARA signatures for blocklist rules and the allowlist rule. We also provide the list of rules without false positives 
    our training set. 

- ### Static Module with XGBoost
    We release the static models deployed inside OBELISK. We provide the XGBoost model files, both for model deployed 
    in OBELISK (with_filters) and the Baseline (no_filters). 

- ### Dynamic Module with Nebula
  We release all the Nebula models trained for each value of $\delta$ used in the ablation study of OBELISK. We also provide the Baseline model, 
  trained on all the dataset. For using is needed to install the original repository at **https://github.com/dtrizna/nebula**

# Attack Interfaces

We provide the attack wrappers for attacking all the surrogates described in the paper. 
When attacking only the static module, we use the xgboost_wrappers or gbdt_transfer when attacking the module of 
Anderson et al. (**https://arxiv.org/abs/1804.04637**) for TM1 and TM3. When attacking subsystems, we use the ai_system wrapper. It is possible to use only the ai_system wrapper
also for models, by specifying the parameter "filtered=False" in the init of the wrapper.
The Anderson et al. model can be found at **https://github.com/endgameinc/malware_evasion_competition/tree/master/models/ember**


# Script to run OBELISK
We provide a simple script to run OBELISK and Baseline inference. The script is located in the root directory, and can be run
with the command:
```bash
python ai_sys_inference.py <sample_path>
```
where `<sample_path>` is the path to the sample to be analyzed. The script will output the result of the inference both for OBELISK and Baseline.

We also provide the script (```attack_on_tm.py```) to initialize the Threat Models with the AI System Wrapper. We provide an example
to run gamma and padding attacks. 