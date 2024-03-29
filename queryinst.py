from typing import List, Tuple, Optional

import numpy as np
from src.resnet import ResidualBlock

import mindspore.nn as nn
from mindspore import Tensor

from src.rpn_head import EmbeddingRPNHead
from src.roi_head import SparseRoIHead
from src.resnet import ResNet
from src.fpn_pt import FPN

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

class QueryInst(nn.Cell):
    """Base class for QueryInst detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self):
        super().__init__()
        
        self.backbone = ResNet(block=ResidualBlock,
                              layer_nums=[3, 4, 6, 3],
                              in_channels=[64, 256, 512, 1024],
                              out_channels=[256, 512, 1024, 2048],
                              weights_update=False)

        self.neck = FPN(in_channels=[256, 512, 1024, 2048],
                          out_channels=256,
                          num_outs=4,
                          add_extra_convs='on_input',
                          start_level=0)
        
        self.rpn_head = EmbeddingRPNHead(num_proposals=100, proposal_feature_channel=256)
        
        self.match_costs_config = [
            dict(type='FocalLossCost', weight=2.0),
            dict(type='BBoxL1Cost', weight=5.0),
            dict(type='IoUCost', iou_mode='giou')
        ]
        
        self.bbox_roi_extractor_config = dict(roi_layer=dict(out_size=7, sample_num=2),
                                 out_channels=256,
                                 featmap_strides=[4, 8, 16, 32])
        self.mask_roi_extractor_config = dict(roi_layer=dict(out_size=14, sample_num=2),
                                        out_channels=256,
                                        featmap_strides=[4, 8, 16, 32])
        self.bbox_head_config = dict(in_channel=256, inner_channel=64, out_channel=256)
        self.mask_head_config = dict(num_convs=4)
        
        self.roi_head = SparseRoIHead(match_costs_config=self.match_costs_config,
                              bbox_roi_extractor=self.bbox_roi_extractor_config,
                              mask_roi_extractor=self.mask_roi_extractor_config,
                              bbox_head=self.bbox_head_config,
                              mask_head=self.mask_head_config)

    def extract_feat(self, batch_inputs):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        x = self.neck(x)
        return x

    def construct(self, batch_inputs: Tensor,
                  batch_data_samples: List[dict]) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[dict]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        rpn_results_list = self.rpn_head(
            x, batch_data_samples)

        roi_outs = self.roi_head(x, rpn_results_list,
                                 batch_data_samples)
        results = roi_outs
        
        return results

