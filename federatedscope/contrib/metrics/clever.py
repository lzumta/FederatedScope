import json
import logging
import os
from json import JSONDecodeError

import numpy as np
from federatedTrust.calculation import get_clever_score
from federatedTrust.utils import get_value_from_path

from federatedscope.register import register_metric

logger = logging.getLogger(__name__)

METRIC_NAME = 'clever'


def compute_clever_metric(ctx, **kwargs):
    test_clever = 0
    if ctx.cur_split == 'test' or ctx.cur_split == 'val':
        with open(os.path.join(os.getcwd(), ctx.cfg['outdir'], 'factsheet.json'), 'r+') as f:
            try:
                factsheet = json.load(f)
                test_clever = get_value_from_path(factsheet, "performance/test_clever")
                if np.isnan(test_clever):
                    x = ctx.data[ctx.cur_split]
                    test_sample = next(iter(x))
                    nb_classes = ctx.cfg['model']['out_channels']
                    lr = ctx.cfg['train']['optimizer']['lr']
                    cfg = {"nb_classes": nb_classes, "lr": lr}
                    value = get_clever_score(test_sample, ctx.model, cfg)
                    test_clever = 1 if value > 1 else value
                    factsheet["performance"]["test_clever"] = test_clever
                    f.seek(0)  # set cursor back to line 0
                    json.dump(factsheet, f)
                    f.truncate()
            except JSONDecodeError as e:
                logger.warning(f"factsheet is invalid")
                logger.error(e)
    return test_clever


def clever_metric(types):
    if METRIC_NAME in types:
        metric_builder = compute_clever_metric
        return METRIC_NAME, metric_builder


register_metric(METRIC_NAME, clever_metric)
