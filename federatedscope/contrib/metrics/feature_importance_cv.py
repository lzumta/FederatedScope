import json
import logging
import os
from json import JSONDecodeError

import numpy as np
from federatedTrust.calculation import get_feature_importance_cv
from federatedTrust.utils import get_value_from_path

from federatedscope.register import register_metric

logger = logging.getLogger(__name__)

METRIC_NAME = 'feature_importance_cv'


def compute_feature_importance_cv_metric(ctx, **kwargs):
    cv = 0
    if ctx.cur_split == 'test' or ctx.cur_split == 'val':
        with open(os.path.join(os.getcwd(), ctx.cfg['outdir'], 'factsheet.json'), 'r+') as f:
            try:
                factsheet = json.load(f)
                cv = get_value_from_path(factsheet, "performance/test_feature_importance_cv")
                if cv == "":
                    x = ctx.data[ctx.cur_split]
                    batch_size = ctx.cfg['data']['batch_size']
                    test_sample = next(iter(x))
                    cfg = {"batch_size": batch_size, "device": ctx.device}
                    value = get_feature_importance_cv(test_sample, ctx.model, cfg)
                    cv = 1 if value > 1 else value
                    factsheet["performance"]["test_feature_importance_cv"] = cv
                    f.seek(0)  # set cursor back to line 0
                    json.dump(factsheet, f)
                    f.truncate()
            except JSONDecodeError as e:
                logger.warning(f"factsheet is invalid")
                logger.error(e)
    return cv


def feature_importance_cv_metric(types):
    if METRIC_NAME in types:
        metric_builder = compute_feature_importance_cv_metric
        return METRIC_NAME, metric_builder


register_metric(METRIC_NAME, feature_importance_cv_metric)
