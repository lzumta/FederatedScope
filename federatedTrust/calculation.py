import logging
import math
import numbers
import os.path
from math import e
from os.path import exists
import pandas as pd
import codecarbon
import numpy as np
import shap
import torch.nn
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u
from scipy.stats import variation
from torch import nn, optim

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

R_L1 = 40
R_L2 = 2
R_LI = 0.1


def get_mapped_score(score_key, score_map):
    """Finds the score by the score_key in the score_map
        :param score_key: the key to look up in the score_map
        :param score_map: the defined score map
        :return: normalized score of [0, 1]
    """
    score = 0
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        keys = [key for key, value in score_map.items()]
        scores = [value for key, value in score_map.items()]
        normalized_scores = get_normalized_scores(scores)
        normalized_score_map = dict(zip(keys, normalized_scores))
        score = normalized_score_map.get(score_key, np.nan)
    return score


def get_normalized_scores(scores):
    normalized = [(x - np.min(scores))/(np.max(scores) - np.min(scores)) for x in scores]
    return normalized


def get_range_score(value, ranges, direction='asc'):
    """Maps the value to a range and gets the score by the range and direction
        :param value: the input score
        :param ranges: the ranges defined
        :param direction: asc means the higher the range the higher the score, desc means otherwise
        :return: normalized score of [0, 1]
    """
    if not (type(value) == int or type(value)== float):
        logger.warning("Input value is not a number")
        logger.warning(f"{value}")
        return 0
    else:
        score = 0
        if ranges is None:
            logger.warning("Score ranges are missing")
        else:
            total_bins = len(ranges) + 1
            bin = np.digitize(value, ranges, right=True)
            score = 1 - (bin / total_bins) if direction == 'desc' else bin / total_bins
        return score


def get_ranked_score(score_key, score_map, direction):
    """Finds the score by the score_key in the score_map and returns the rank of the score
        :param score_key: the key to look up in the score_map
        :param score_map: the score map defined
        :param direction: asc means the higher the range the higher the score, desc means otherwise
        :return: normalized score of [0, 1]
    """
    score = 0
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        sorted_scores = sorted(score_map.items(),
                               key=lambda item: item[1],
                               reverse=direction == 'desc')
        sorted_score_map = dict(sorted_scores)
        for index, key in enumerate(sorted_score_map):
            if key == score_key:
                score = (index + 1) / len(sorted_score_map)
    return score


def get_true_score(value, direction):
    """Returns the negative of the value if direction is 'desc', otherwise returns value
        :param value: the input value
        :param direction: asc means the higher the range the higher the score, desc means otherwise
        :return: object
    """
    if value is True:
        return 1
    elif value is False:
        return 0
    else:
        if not(type(value) == int or type(value) == float):
            logger.warning("Input value is not a number")
            logger.warning(f"{value}.")
            return 0
        else:
            if direction == 'desc':
                return 1 - value
            else:
                return value


def get_value(value):
    """Returns the value
        :param value: the input value
        :return: the value object
    """
    return value


def check_properties(*args):
    """Check if all the arguments have values
        :param args: all the arguments
        :return: the mean of the binary array
    """
    result = map(lambda x: x is not None and x != "", args)
    return np.mean(list(result))


def get_cv(list=None, std=None, mean=None):
    """Calculates the coefficient of variation
       :param std: the standard deviation
       :param mean: the mean
       :return: coefficient of variation of the dataset
   """
    if std is not None and mean is not None:
        return std / mean

    if list is not None:
        return np.std(list) / np.mean(list)

    return 0


def get_global_privacy_risk(dp, epsilon, n):
    """Calculates the global privacy risk by epsilon and the number of clients
       :param dp: True or False
       :param epsilon: the epsilon value
       :param n: number of clients
       :return: the global privacy risk
    """
    if dp is True and isinstance(epsilon, numbers.Number):
        return 1 / (1 + (n - 1) * math.pow(e, -epsilon))
    else:
        return 1


def get_feature_importance_cv(test_sample, model, cfg):
    """Calculates feature importance coefficient of variation
       :param test_sample: one test sample
       :param model: the model
       :param cfg: configs
       :return: the coefficient of variation of the feature importance scores, [0, 1]
    """
    try:
        cv = 0
        batch_size = cfg['batch_size']
        device = cfg['device']
        if isinstance(model, torch.nn.Module):
            batched_data, _ = test_sample

            n = batch_size
            m = math.floor(0.8 * n)

            background = batched_data[:m].to(device)
            test_data = batched_data[m:n].to(device)

            e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(test_data)
            if shap_values is not None and len(shap_values) > 0:
                sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
                abs_sums = np.absolute(sums)
                cv = variation(abs_sums)
    except Exception as e:
        logger.warning("Could not compute feature importance CV with shap")
        cv = 0
    if math.isnan(cv):
        cv = 0
    return cv


def get_clever_score(test_sample, model, cfg):
    """Calculates the CLEVER score
       :param test_sample: one test sample
       :param model: the model
       :param cfg: configs
       :return: the CLEVER score of type number
    """
    nb_classes = cfg['nb_classes']
    lr = cfg['lr']
    images, _ = test_sample
    background = images[-1]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 255.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=nb_classes,
    )
    score_untargeted = clever_u(classifier, background.numpy(), 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
    return score_untargeted

def get_scaled_score(value, scale:list,  direction:str):
    """Maps a score of a specific scale into the scale between zero and one
        :param value: int or float: the raw value of the metric
        :param scale: list containing the minimum and maximum value the value can fall inbetween
        :param direction: asc means the higher the range the higher the score, desc means otherwise
        :return: normalized score of [0, 1]
    """
    score = 0
    try:
        value_min, value_max = scale[0], scale[1]
    except Exception as e:
        logger.warning("Score minimum or score maximum is missing. The minimum has been set to 0 and the maximum to 1")
        value_min, value_max = 0,1
    if not value:
        logger.warning("Score value is missing. Set value to zero")
    else:
        low, high = 0, 1
        if value >= value_max:
            score = 1
        elif value <= value_min:
            score = 0
        else:

            diff = value_max - value_min
            diffScale = high - low
            score =  ((float(value) - value_min) * (float(diffScale) / diff) + low)
        if direction == 'desc':
            score = high - score

    return score


def stop_emissionstracking_and_save(tracker: codecarbon.EmissionsTracker, outdir: str, emissions_file: str, role: str, workload: str, sample_size: int = 0):
    """ Stops emissions tracking object from CodeCarbon and saves relevant information to emissions.csv file
    :param tracker: codecarbon.EmissionsTacker: the emissions tracker object holding information
    :param outdir: str: the path of the output directory of the experiment
    :param emissions_file: str: the path to the emissions file
    :param role: str: either client or server depending on the role
    :param workload: str: either aggregation or training depending on the workload
    :param sample_size: int: the number of samples used for training, if aggregation 0
    """
    tracker.stop()

    emissions_file = os.path.join(outdir, emissions_file)

    if exists(emissions_file):
        df = pd.read_csv(emissions_file)
    else:
        df = pd.DataFrame(columns=["role", "energy_grid", "emissions", "workload", 'CPU_model', 'GPU_model'])
    try:
        energy_grid = (tracker.final_emissions_data.emissions / tracker.final_emissions_data.energy_consumed) * 1000
        df = pd.concat([df, pd.DataFrame({'role': role, 'energy_grid': [energy_grid],
                                          'emissions': [tracker.final_emissions_data.emissions], 'workload': workload,
                                          'CPU_model': tracker.final_emissions_data.cpu_model if tracker.final_emissions_data.cpu_model else "None",
                                          'GPU_model': tracker.final_emissions_data.gpu_model if tracker.final_emissions_data.gpu_model else "None",
                                          'CPU_used': True if tracker.final_emissions_data.cpu_energy else False,
                                         'GPU_used': True if tracker.final_emissions_data.gpu_energy else False,
                                          'energy_consumed': tracker.final_emissions_data.energy_consumed,
                                          'sample_size': sample_size})],
                       ignore_index=True)
        df.to_csv(emissions_file, encoding='utf-8', index=False)
    except Exception as e:
        logger.warning(e)


