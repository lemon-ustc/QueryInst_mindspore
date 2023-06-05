import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import mindspore.nn as nn

from .fun_assi import box_cxcywh_to_xyxy
from .fun_assi import giou


def softmax(arr, axis=None):
    """softmax"""
    return np.exp(arr) / np.sum(np.exp(arr), axis=axis, keepdims=True)


class HungarianAssigner(nn.Cell):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, 
                 cls_weight = 1.0,
                 reg_weight = 1.0,
                 iou_weight = 1.0):
        """Creates the matcher

        Params:
            cls_weight: This is the relative weight of the classification error in the matching cost
            reg_weight: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            iou_weight: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cls_weight
        self.cost_bbox = reg_weight
        self.cost_giou = iou_weight
        # assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
            #    img_meta,
            #    gt_bboxes_ignore=None,
               eps=1e-7):
     
    # def assign(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # bs, num_queries = outputs["pred_logits"].shape[:2]
        bs, num_queries = cls_pred.shape[:2]
        # pred_logits = outputs['pred_logits']
        # pred_boxes = outputs['pred_boxes']
        pred_boxes = bbox_pred
        pred_logits = cls_pred
        # We reshape to compute the cost matrices in a batch
        # out_prob [batch_size * num_queries, num_classes]
        out_prob = softmax(pred_logits.reshape(-1, pred_logits.shape[-1]), -1)
        # out_bbox [batch_size * num_queries, 4]
        out_bbox = pred_boxes.reshape(-1, pred_boxes.shape[-1])

        # Also concat the target labels and boxes
        tgt_ids = np.concatenate(gt_labels)
        tgt_bbox = np.concatenate(gt_bboxes)
        # tgt_ids = np.concatenate([v["labels"] for v in targets])
        # tgt_bbox = np.concatenate([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = cdist(out_bbox, tgt_bbox, metric='minkowski', p=1)

        # Compute the giou cost between boxes
        cost_giou = -giou(box_cxcywh_to_xyxy(out_bbox),
                            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        final_cost_matrix = (self.cost_bbox * cost_bbox +
                             self.cost_class * cost_class + 
                             self.cost_giou * cost_giou)
        final_cost_matrix = final_cost_matrix.reshape(bs, num_queries, -1)

        sizes = np.cumsum(len(gt_bboxes))
        # sizes = np.cumsum([len(v["boxes"]) for v in targets])
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(np.split(final_cost_matrix, sizes, -1)[:-1])
        ]
        return [(i, j) for i, j in indices]


def build_assigner(config):
    """build hungarian matcher"""
    return HungarianAssigner(cost_class=config.set_cost_class,
                            cost_bbox=config.set_cost_bbox,
                            cost_giou=config.set_cost_giou)
