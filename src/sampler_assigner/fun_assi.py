import numpy as np

def box_cxcywh_to_xyxy(x):
    """box cxcywh to xyxy"""
    x_c, y_c, w, h = np.array_split(x, 4, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1).squeeze(-2)

def box_area(box):
    """box area"""
    return (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])

def box_iou(boxes1, boxes2):
    """box iou"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union, inter, wh


def giou(boxes1, boxes2, calc_grad=False):

    # degenerate boxes gives inf / nan results
    # so do an early check
    iou, union, inter, wh1 = box_iou(boxes1, boxes2)

    lt = np.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / area

    if calc_grad:
        n_boxes = boxes1.shape[0]
        ddiagres_dres = -np.eye(n_boxes)
        dres_diou = ddiagres_dres
        dres_darea = ddiagres_dres * (-union / area ** 2)
        diou_dunion = dres_diou * (-inter / union ** 2)
        dres_dunion = diou_dunion + ddiagres_dres * (1 / area)
        dres_dinter = - dres_dunion + dres_diou / union

        src1 = dres_dwh(dres_dinter, wh1, True, boxes1, boxes2)

        dunion_darea1 = dres_dunion
        src2 = dunion_darea1.sum(axis=1, keepdims=True) * area_box_grad(boxes1)

        src3 = dres_dwh(dres_darea, wh, False, boxes1, boxes2)
        src_grad = grad_xywh_to_cxcy(src1 + src2 + src3) / n_boxes

        return giou, src_grad
    return giou
#giou_fun
def area_box_grad(box):
    """box area grad"""
    return np.stack([
        box[:, 1] - box[:, 3],
        box[:, 0] - box[:, 2],
        box[:, 3] - box[:, 1],
        box[:, 2] - box[:, 0]
    ], axis=1)


def grad_xywh_to_cxcy(arr):
    """xywh to cxcy grad"""
    return np.stack([
        arr[:, 0] + arr[:, 2],
        arr[:, 1] + arr[:, 3],
        (-arr[:, 0] + arr[:, 2]) / 2,
        (-arr[:, 1] + arr[:, 3]) / 2
    ], axis=1)


def dres_dwh(dres_df, wh, is_maxmin,
             src_boxes_cxcy, target_boxes_cxcy):
    """dres/dwh"""
    d_dwh = np.stack([dres_df * wh[:, :, 1], dres_df * wh[:, :, 0]], axis=2)
    if is_maxmin:
        dwh_dmax = -d_dwh * (wh > 0.).astype(np.int32)
        dwh_dmin = d_dwh * (wh > 0.).astype(np.int32)
        max_grad_src = (src_boxes_cxcy[:, :2] > target_boxes_cxcy[:, :2]).astype(np.int32)
        min_grad_src = (src_boxes_cxcy[:, 2:] < target_boxes_cxcy[:, 2:]).astype(np.int32)
        src_grad = np.concatenate([
            (dwh_dmax * max_grad_src).sum(axis=1),
            (dwh_dmin * min_grad_src).sum(axis=1)
        ], axis=1)
    else:
        dwh_dmax = d_dwh * (wh > 0.).astype(np.int32)
        dwh_dmin = -d_dwh * (wh > 0.).astype(np.int32)
        max_grad_src = (src_boxes_cxcy[:, 2:] > target_boxes_cxcy[:, 2:]).astype(np.int32)
        min_grad_src = (src_boxes_cxcy[:, :2] < target_boxes_cxcy[:, :2]).astype(np.int32)
        src_grad = np.concatenate([
            (dwh_dmin * min_grad_src).sum(axis=1),
            (dwh_dmax * max_grad_src).sum(axis=1)
        ], axis=1)

    return src_grad