import torch
import torch.utils.tensorboard
import numpy as np
# import model.Darknet53
from model import Darknet53
from model.Darknet53 import darknet53
from model.Darknet53 import  Res_unit
import utils

from torch import nn

class YoloDetection(nn.Module):
    def __init__(self, anchors, image_size: int, num_classes: int):
        super(YoloDetection, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)  # 앵커 박스의 개수
        self.num_classes = num_classes
        self.image_size = image_size
        self.mse_loss = nn.MSELoss()  # Mean Squared Error
        self.bce_loss = nn.BCELoss()  # Binary-cross-Entropy
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.no_obj_scale = 100
        self.metrics = {}

    def forward(self, x, targets):
        device = torch.device('cuda' if x.is_cuda else 'cpu')

        num_batches = x.size(0)  # 1
        grid_size = x.size(2)

        # 출력 형태 변환
        prediction = (x.view(num_batches, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                      .permute(0, 1, 3, 4, 2).contiguous()
                      )  # [1,3,13,13,85] , [1,3,26,26,85], [1,3,52,52,85]

        cx = torch.sigmoid(prediction[..., 0])  # 예측하려는 앵커 박스의 Center x , [1,3,13,13], [1,3,26,26], [1,3,52,52]
        cy = torch.sigmoid(prediction[..., 1])  # 예측하려는 앵커 박스의 Center y    [1,3,13,13], [1,3,26,26], [1,3,52,52]
        w = prediction[..., 2]  # ground truth의 너비   [1,3,13,13], [1,3,26,26], [1,3,52,52]
        h = prediction[..., 3]  # ground truth의 높이    [1,3,13,13], [1,3,26,26], [1,3,52,52]
        pred_conf = torch.sigmoid(prediction[..., 4])  # confidence score   [1,3,13,13], [1,3,26,26], [1,3,52,52]
        pred_cls = torch.sigmoid(prediction[..., 5:])  # class   [1,3,13,13,80], [1,3,26,26,80], [1,3,52,52,80]

        # 각각의 grid offset 계산
        stride = self.image_size / grid_size  # 416/13 -> 32, 416/26 ->16, 416/52 -> 8
        grid_x = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).view(
            [1, 1, grid_size, grid_size])  # [1,1,13,13] [1,1,26,26] [1,1,52,52]
        grid_y = torch.arange(grid_size, dtype=torch.float, device=device).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors],
                                         dtype=torch.float, device=device)
        anchor_w = scaled_anchors[:, 0].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1].view((1, self.num_anchors, 1, 1))

        pred_boxes = torch.zeros_like(prediction[..., :4], device=device)  # [1,3,13,13,4], [1,3,26,26,4], [1,3,52,52,4]
        pred_boxes[..., 0] = cx + grid_x  # cx는 앵커박스 중심으로 그리드셀 안에서 어느 위치에 있는지 알려주고, grid_x는 어떤 그리드 셀인지 알려줌
        pred_boxes[..., 1] = cy + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        pred = (pred_boxes.view(num_batches, -1, 4) * stride,  # 그리드 셀에서의 좌표에서 stride를 곱해줘서 이미지 전체에서의 좌표가 나옴
                pred_conf.view(num_batches, -1, 1),
                pred_cls.view(num_batches, -1, self.num_classes))

        output = torch.cat(pred, -1)  # [1,507,85]

        if targets is None:
            return output, 0

        else:
            iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = utils.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchor=scaled_anchors,
                ignore_thres=self.ignore_thres,
                device=device)

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        loss_layer = loss_bbox + loss_conf + loss_cls

        return loss_layer, output


class yolov3(nn.Module):
    def __init__(self, image_size: int, num_classes: int):
        super(yolov3, self).__init__()


        anchors = {'anchor_52': [(10, 13), (16, 30), (33, 23)],
                   'anchor_26': [(30, 61), (62, 45), (59, 119)],
                   'anchor_13': [(116, 90), (156, 198), (373, 326)]}
        self.darknet53 = darknet53()

        self.DBL5_1 = self.DBL5(1024, 512)
        self.dbl_conv1 = self.DBL_conv(512, 255)  # 255 : 3 * (4 + 1 + num_classes(80))
        self.yolo_layer1 = YoloDetection(anchors['anchor_13'], image_size, num_classes)

        self.DBL5_2 = self.DBL5(768, 256)
        self.dbl_conv2 = self.DBL_conv(256, 255)
        self.yolo_layer2 = YoloDetection(anchors['anchor_26'], image_size, num_classes)

        self.DBL5_3 = self.DBL5(384, 128)
        self.dbl_conv3 = self.DBL_conv(128, 255)
        self.yolo_layer3 = YoloDetection(anchors['anchor_52'], image_size, num_classes)

        self.db_samp1 = self.DBL_Upsampling(512, 256, scale_factor=2)

        self.db_samp2 = self.DBL_Upsampling(256, 128, scale_factor=2)

        self.yolo_layers = [self.yolo_layer1, self.yolo_layer2, self.yolo_layer3]

    def DBL5(self, in_c, out_c):
        double_c = out_c * 2
        dbl5 = nn.Sequential(
            Darknet53.DBL(in_c, out_c, kernel_size=1, padding=0),
            Darknet53.DBL(out_c, double_c, kernel_size=3),
            Darknet53.DBL(double_c, out_c, kernel_size=1, padding=0),
            Darknet53.DBL(out_c, double_c, kernel_size=3),
            Darknet53.DBL(double_c, out_c, kernel_size=1, padding=0)
        )
        return dbl5

    def DBL_Upsampling(self, in_c, out_c, scale_factor):
        db_samp = nn.Sequential(
            Darknet53.DBL(in_c, out_c, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='nearest')
        )
        return db_samp

    def DBL_conv(self, in_c, out_c):
        db_conv = nn.Sequential(
            Darknet53.DBL(in_c, in_c * 2, kernel_size=3),
            nn.Conv2d(in_c * 2, out_c, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return db_conv

    def forward(self, x, targets=None):
        loss = 0
        outs = self.darknet53.forward(x)

        # print("rrrrrrrrrrr:",self.r3.shape)
        out = self.DBL5_1(outs[2]) # outs[2] = [1,1024, 13,13]
        Feature_1 = self.dbl_conv1(out)

        anchor13, loss_layer1 = self.yolo_layer1(Feature_1, targets)
        print('anchor13', anchor13.shape)

        out1 = self.db_samp1(out)
        print(out1.shape)
        out2 = torch.cat([out1, outs[1]], dim=1)
        out3 = self.DBL5_2(out2)
        Feature_2 = self.dbl_conv2(out3)

        anchor26, loss_layer2 = self.yolo_layer2(Feature_2, targets)
        print('anchor26', anchor26.shape)

        out4 = self.db_samp2(out3)
        out5 = torch.cat([out4, outs[0]], dim=1)
        out6 = self.DBL5_3(out5)
        Feature_3 = self.dbl_conv3(out6)

        anchor52, loss_layer3 = self.yolo_layer3(Feature_3, targets)
        print('anchor52 : ',anchor52.shape)

        # anchor box 합치기
        yolo_output = [anchor13, anchor26, anchor52]
        yolo_output = torch.cat(yolo_output, 1).detach() # 인덱스 1번째 차원으로 함치기
        print(yolo_output.shape) # [1, 10647, 85]

        # 최종 loss
        loss = loss_layer1 + loss_layer2 + loss_layer3

        return yolo_output if targets is None else loss, yolo_output

if __name__ == '__main__':
    model = yolov3(416,80)
    img = torch.rand(1, 3, 416, 416)
    y = model(img)

#
# img = torch.rand(1, 3, 416, 416)
# # model = yolov3(416,80)
# loss, yolo_outputs =yolov3(416,80).forward(img)#,3
# print(yolo_outputs.shape)
# yolov3.forward(img)

# img = torch.randn([416,416])
