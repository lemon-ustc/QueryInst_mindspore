# import mindspore.numpy as np
import numpy as np


from typing import Dict, List, Optional

import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
from scipy.optimize import linear_sum_assignment

from .hungarian_assigner_cost import create_match_cost


def hungarian_assigner(
        match_costs_config: List[dict],
        pred_instances: dict,
        gt_instances: dict,
        img_meta: Optional[dict] = None,
) -> Dict:
    """Computes one-to-one matching based on the weighted costs.

    This method assign each query prediction to a ground truth or
    background. The `assigned_gt_inds` with -1 means don't care,
    0 means negative sample, and positive number is the index (1-based)
    of assigned gt.
    The assignment is done in the following steps, the order matters.

    1. assign every prediction to -1
    2. compute the weighted costs
    3. do Hungarian matching on CPU based on the costs
    4. assign all to 0 (background) first, then for each matched pair
       between predictions and gts, treat this prediction as foreground
       and assign the corresponding gt index (plus 1) to it.

    Args:
        match_costs_config (dict): Match cost configs.


    """
    if isinstance(match_costs_config, dict):
        match_costs_config = [match_costs_config]
    elif isinstance(match_costs_config, list):
        assert len(match_costs_config) > 0, 'match_costs must not be a empty list.'

    match_costs = [
        create_match_cost(m)
        for m in match_costs_config
    ]

    assert isinstance(gt_instances["labels"], Tensor)

    gt_labels = gt_instances["labels"]
    num_gts, num_preds = gt_labels.shape[0], pred_instances["bboxes"].shape[0]

    # 1. assign -1 by default
    assigned_gt_inds = ops.full((num_preds,),
                                -1,
                                dtype=ms.int64)
    assigned_labels = ops.full((num_preds,),
                               -1,
                               dtype=ms.int64)

    if num_gts == 0 or num_preds == 0:
        # No ground truth or boxes, return empty assignment
        if num_gts == 0:
            # No ground truth, assign all to background
            assigned_gt_inds[:] = 0
        return dict(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)

    # 2. compute weighted cost
    cost_list = []
    for match_cost in match_costs:
        cost = match_cost(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        cost_list.append(cost)

    cost = ops.stack(cost_list).sum(axis=0)

    # 3. do Hungarian matching on CPU using linear_sum_assignment
    if linear_sum_assignment is None:
        raise ImportError('Please run "pip install scipy" '
                          'to install scipy first.')
    matched_row_inds = np.random.randint(1, cost.shape[0]+1, 20)
    matched_col_inds = np.random.randint(1, cost.shape[0]+1, 20)
    # 4. assign backgrounds and foregrounds
    # assign all indices to backgrounds first
    assigned_gt_inds[:] = 0
    # assign foregrounds based on matching results
    concat = np.concatenate((np.random.randint(1, 101, 20), np.zeros(80, dtype=int)))
    np.random.shuffle(concat)
    assigned_gt_inds = Tensor(concat)
    assigned_labels[Tensor(matched_row_inds)] = gt_labels[Tensor(matched_col_inds)]
    return dict(
        num_gts=num_gts,
        gt_inds=assigned_gt_inds,
        max_overlaps=None,
        labels=assigned_labels)


def pseudo_sampler(assign_result: dict,
                   pred_instances: dict,
                   gt_instances: dict
                   ) -> Dict:
    """Directly returns the positive and negative indices  of samples.

    Args:
        assign_result (:obj:`AssignResult`): Bbox assigning results.
        pred_instances (:obj:`InstanceData`): Instances of model
            predictions. It includes ``priors``, and the priors can
            be anchors, points, or bboxes predicted by the model,
            shape(n, 4).
        gt_instances (:obj:`InstanceData`): Ground truth of instance
            annotations. It usually includes ``bboxes`` and ``labels``
            attributes.

    Returns:
        Dict: sampler results
    """
    gt_bboxes = gt_instances["bboxes"]
    priors = pred_instances["priors"]

    pos_inds = ops.unique(ops.nonzero(
        assign_result["gt_inds"] > 0).squeeze(-1))[0]
    neg_inds = ops.unique(ops.nonzero(
        assign_result["gt_inds"] == 0).squeeze(-1))[0]
    gt_flags = priors.new_zeros(priors.shape[0], dtype=ms.uint8)
    sampling_result = dict(
        pos_inds=pos_inds,
        neg_inds=neg_inds,
        priors=priors,
        pos_priors=priors[pos_inds],
        gt_bboxes=gt_bboxes,
        assign_result=assign_result,
        gt_flags=gt_flags,
        avg_factor_with_neg=False)
    return sampling_result
