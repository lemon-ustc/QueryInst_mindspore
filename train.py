"""train Queryinst"""

import time
import os

from src.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num

from dataset import data_to_mindrecord_byte_image, create_queryinst_dataset
from src.lr_schedule import dynamic_lr
from queryinst import QueryInst


import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

class WithLossCell(nn.Cell):
    def __init__(self, net, criterion):
        super(WithLossCell, self).__init__()
        self.net = net
        self.criterion = criterion

    def construct(self, data):
        img = data[0]
        bboxes = data[2]
        labels = data[3]
        seg = data[5]
        batch_gt_instances = {"bboxes": bboxes.squeeze(0),
                                "labels": labels.squeeze(0)}
        batch_img_metas = {"img_shape": [768, 1280]}
        batch_data_samples = [{"gt_instances": batch_gt_instances, "metainfo": batch_img_metas}]
        pre = self.net(img, batch_data_samples)
        pred = pre[-1][1]['mask_preds'].squeeze()
        loss = self.criterion(pred, seg[0, 0, :, :])
        return loss

def train():
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=get_device_id())
    
    if not config.do_eval and config.run_distribute:
        init()
        rank = get_rank()
        device_target == 'Ascend'
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    print(f"\n[{rank}] - rank id of process")
    
    """ create dataset """
    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "Queryinst.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file + ".db"):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if config.dataset == "coco":
            if os.path.isdir(config.coco_root):
                if not os.path.exists(config.coco_root):
                    print("Please make sure config:coco_root is valid.")
                    raise ValueError(config.coco_root)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                if not os.path.exists(config.image_dir):
                    print("Please make sure config:image_dir is valid.")
                    raise ValueError(config.image_dir)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image(config, "other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_queryinst_dataset(mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank)

    dataset_size = dataset.get_dataset_size()
    print("total images num: ", dataset_size)
    print("Create dataset done!")
    

    net = QueryInst()

    lr = Tensor(dynamic_lr(config, rank_size=device_num, start_steps=config.pretrain_epoch_size * dataset_size),
                mstype.float32)
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                    weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    model = QueryInst()
    
    net_with_loss = WithLossCell(model, nn.DiceLoss())
    
    net = nn.TrainOneStepCell(net_with_loss, opt)
    
    iteration = 0
    for epoch in range(config.total_epoch):    
        for index, data in enumerate(dataset): 
            loss = net(data)
            iteration += 1
            print("iteration ", iteration, "loss ", loss)
      
if __name__ == '__main__':
    train()
