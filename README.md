# FederatedTrust integrated into FederatedScope
An algorithm to calculate trustworthiness score of the federated learning framework `FederateScope`. The current methods provided are written based on how `FederatedScope` works. For future work, more methods can be created for other types of federated learning frameworks.
The algorithm computes the trustworthiness score of the federated learning system with the seven pillars: privacy, fairness, accountability, explainability, robustness, federation (also called architectural soundness) and sustainability with over 20 notions and over 30 metrics.
This prototype encorporates all the seven requirements for trustworthy AI defined by the 

## Used Tools
This algorithmic prototype extends the first FederatedTrust version: https://github.com/ningxie1991/FederatedTrust/ with a trust pillar called sustainability and is used in combination with FederatedScope v.0.2.0: https://github.com/alibaba/FederatedScope/releases/tag/v0.2.0.
A big thanks and credits go to the developers of these two preliminary works that enable me to develop an extended version of FederatedTrust included into FederatedScope.
## Installation
1. Clone the repository with git clone
```
git clone URL
```
2. Change the directory
```
cd FederatedScope
```
3. Install the dependencies
```
pip3 install -r requirements.txt
```
## Example use-case
1. Change the directory to FederatedTrust
```
cd federatedTrust
```
2. Run a simple federation with the example configurations
```
python ../federatedScope/main.py --cfg configs/example_config.yaml
```
## Example output
### Command line output
```
+-------------------------+--------+
| trust_score             |   0.62 |
|-------------------------+--------|
| robustness              |   0.22 |
| privacy                 |   0.49 |
| fairness                |   0.33 |
| explainability          |   0.27 |
| accountability          |   0.73 |
| architectural_soundness |   0.78 |
| sustainability          |   0.77 |
+-------------------------+--------+
```
### federatedtrust_results.json
A file containing detailed results of the trustworthiness evaluation will be stored in the federatedtrust_results.json file in the path federatedTrust/experiments/evaluation/subexperiment{...}/
```
{"trust_score": 0.44, 
    "pillars": 
        [{"robustness": 
            {"score": 0.33, 
            "notions": 
                [{"resilience_to_attacks": 
                    {"score": 0.02, 
                    "metrics": 
                        [{"certified_robustness": {"score": 0.02}}]}},
                {"algorithm_robustness": 
                    {"score": 0.51, 
                    "metrics": 
                        [{"performance": {"score": 0.02}}, 
                        {"personalization": {"score": 1}}]}},
                {"client_reliability": 
                    {"score": 0.45, 
                        "metrics": 
                            [{"scale": {"score": 0.45}}]}}]}}, 
           {"privacy": 
                {"score": 0.55, 
                "notions": 
                    [{"technique": 
                        {"score": 1.0, 
                        "metrics": 
                            [{"differential_privacy": {"score": 1}}]}},
                    {"uncertainty": 
                        {"score": 0.65, 
                        "metrics": 
                            [{"entropy": {"score": 0.65}}]}}, 
                    {"indistinguishability": 
                        {"score": 0.0, 
                        "metrics": 
                            [{"global_privacy_risk": {"score": 0.0}}]}}]}}, 
           {"fairness": 
                {"score": 0.26, 
                "notions": 
                [{"selection_fairness": 
                    {"score": 0.78, 
                    "metrics": 
                        [{"selection_variation": {"score": 0.78}}]}},
                {"performance_fairness": 
                    {"score": 0.35, 
                    "metrics": 
                        [{"accuracy_variation": {"score": 0.35}}]}}, 
                {"class_distribution": 
                    {"score": 0.0, 
                    "metrics": 
                        [{"class_imbalance": {"score": 0}}]}}]}},
           {"explainability": 
                {"score": 0.9, 
                "notions": 
                    [{"interpretability": 
                    {"score": 0.8, 
                    "metrics": 
                        [{"algorithmic_transparency": {"score": 0.0}}, 
                        {"model_size": {"score": 0.8}}]}},
                    {"post_hoc_methods": 
                    {"score": 1, 
                    "metrics": 
                        [{"feature_importance": {"score": 1}}]}}]}}, 
           {"accountability": 
                {"score": 0.73, 
                "notions": 
                    [{"factsheet_completeness": 
                    {"score": 0.733, 
                    "metrics": 
                        [{"project_specs": {"score": 0.33}}, 
                        {"participants": {"score": 1.0}}, 
                        {"data": {"score": 1.0}}, 
                        {"configuration": {"score": 1.0}},
                        {"performance": {"score": 1.0}}, 
                        {"fairness": {"score": 1.0}}, 
                        {"system": {"score": 1.0}}]}}]}}, 
           {"architectural_soundness": 
                {"score": 0.78, 
                "notions": 
                    [{"client_management": 
                    {"score": 1.0, 
                    "metrics": 
                        [{"client_selector": {"score": 1.0}}]}}, 
                    {"optimization": 
                    {"score": 0.57, 
                    "metrics": 
                        [{"algorithm": {"score": 0.57}}]}}]}}, 
           {"sustainability": 
                {"score": 0.25, 
                "notions": 
                    [{"energy_source": 
                    {"score": 0.11, 
                    "metrics": 
                        [{"carbon_intensity_clients": {"score": 0.11}},
                        {"carbon_intensity_server": {"score": 0.11}}]}}, 
                    {"hardware_efficiency": 
                    {"score": 0.28, 
                    "metrics": 
                        [{"avg_power_performance_clients": {"score": 0.28}}, 
                        {"avg_power_performance_server": {"score": 0.28}}]}}, 
                    {"federation_complexity": 
                        {"score": 0.48666664719999997, 
                        "metrics": 
                            [{"number_of_training_rounds": {"score": 0.17}}, 
                            {"avg_model_size": {"score": 1.0}},
                            {"client_selection_rate": {"score": 0.78}}, 
                            {"number_of_clients": {"score": 0.14}}, 
                            {"local_training_rounds": {"score": 0}}, 
                            {"avg_dataset_size": {"score": 0.83}}]}}]}}]}
```
## Configuration
To configure the federation, set the according parameters in the config file at federatedTrust/configs/example_config.yaml. Other configuration files can be added and passed as a parameter --cfg in the execution as well.

## Parametrization
Running big federations with lots of training rounds and clients can be time-consuming and costly, thus this version of FederatedTrust provides the possibility to parametrize the federation to compute the trust score. Parametrization means setting different metrics that are used to compute the trust score directly instead of actually running the federation.
With this feature, different configurations can be explored and their according trust score observed without actually running them. To parametrize the federation, set the wished metrics in the factsheet_template.json file at federatedTrust/configs. 

## Dependencies
- adversarial-robustness-toolbox == 1.14.1
- codecarbon == 2.2.1
- dotmap == 1.3.30
- numpy==1.22.4
- scikit-learn==1.1.3
- scipy==1.7.3
- pandas==2.0.1
- hashids == 1.3.1
- grpcio==1.55.0
- grpcio-tools
- protobuf == 3.20.3
- pympler == 1.0.1
- pyyaml==6.0
- fvcore
- iopath
- wandb==0.15.3
- shap == 0.41.0
- tabulate == 0.9.0
- tensorboard
- tensorboardX
- tensorflow == 2.12.0
- torch == 2.0.1
