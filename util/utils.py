import torch
from torch import nn

def parse_data_config(path: str):
    """데이터셋 설정 파일을 parse 한다"""
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip() # 좌 우 공백을 분리해준다
        key, value = line.split('=') # '='을 기준으로 분리해서 리스트로 반환
        options[key.strip()] = value.strip() # ???
    return options

def load_classes(path: str):
    """클래스 이름을 로드한다"""
    with open(path, 'r') as f:
        names = f.readlines()
    for i, name in enumerate(names):
        names[i] = name.strip()
    return names



def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, device):
    nB = pred_boxes.size(0)  # batch size
    nA = pred_boxes.size(1)  # anchor 개수 : 3
    nC = pred_cls.size(-1)  # 클래스 개수 : 80
    nG = pred_boxes.size(2)  # grid_size

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)  # 물체가 있을 경우 1이 된다
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)  # 물체가 있을 경우 0이 된다
    class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  # ground truth 박스에서 오프셋과 중심점의 변회량
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float, device=device)  # 정답 클래스

    # Convert to potition relative to box
    target_boxes = target[:,2:6] * nG  # 타켓이 1x1 박스안에 함축해서 들어가있고, 그리드 사이즈를 곱해줘서 타겟 박스를 정의 (그리드값을 곱해줘서 실제 피쳐맵 크기로 볼수 있게해줌)
    gxy = target_boxes[:, :2]  # gxy는 타켓 박스(griund truth) 의 x,y 좌표
    gwh = target_boxes[:, 2:]  # gwh는 타겟 박스(ground truth) 의 가로길이와 세로길이

    # iou 값이 가장 큰 anchor box를 찾는다
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # anchors에는 앵커박스의 가로, 세로의 길이
    _, best_ious_idx = ious.max(0)  # 값,인덱스   # best_ious_idx에는 ground truth와 iou가 가장 높은 엥커박스의 인덱스가 들어감

    # 타겟 분리
    b, target_labels = target[:, :2].long().t()  # b가 배치, target_labels가 물체중 몇번째 인지 나타냄
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    print(gw.shape)
    gi, gj = gxy.long().t()  # ground truth 박스의 인덱스, 정수로 변환(왼쪽 상단의 모서리 좌표)

    # set masks
    obj_mask[
        b, best_ious_idx, gj, gi] = 1  # 물체가 있는 곳을 1로 만들어줌  # 13x13 이 있을 때  ious가 가장 높은 앵커박스가 있는 셀에 물차가 있을 때 1로 만들어준다
    noobj_mask[b, best_ious_idx, gj, gi] = 0  # 물체가 있는 곳을 0으로 만들어줌

    # set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):  # 앵커 박스의 iou와 각각의 인텍스로 반환
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[
            i]] = 0  # ignore_thres보다 iou값이 높은 앵커박스에 대해서 앵커박스에 해당하는 셀의 값을 0으로 만들어줌

    # ground truth 좌표의 변화량 구하기(offset)
    # Coordinates
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_ious_idx, gj, gi, target_labels] = 1  # 셀

    # Compute label correctness and iou at best anchor
    class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, tx, ty, tw, th, tcls, tconf


def xywh2xyxy(x):
    # x는 [x,y,w,h] 형태
    y = x.new(x.shape)  # new는 복사를 의미
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y  # 앵커박스의 왼쪽 상단 좌표와, 오른쪽 하단 좌표가 출력된다.


def bbox_wh_iou(wh1, wh2):
    print("wh1:", wh1)  # anchor
    print("wh2:", wh2)  # gwh
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    # 2개의 box에 대한 iou를 구하고 return
    # x1y1x2y2가 False면 좌상단 우하단으로 바꿔줌
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] + box1[:, 2] / 2
        b1_x1, b1_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b1_y1, b1_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 2] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 교차되는 직사각형의 죄표를 얻는다
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.max(b1_x2, b2_x2)
    inter_y2 = torch.max(b1_y2, b2_y2)

    # clamp는 min 혹은 max의 범주에 해당되도록 값을 변경하는 것을 의미한다
    # 예를들면 2,3,5가 있을 때 min = 4라고 한다면 최소가 4가 되도록 이하의 값들을 교체한다
    inter_area = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)

    # union
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def nms(prediction, conf_thres, nms_thres):
    # conf_thres 보다 낮은 개체 confidence score로 detection을 제거하고, 추가 탐지를 필터링하기 위해 nms 수행
    # 모양 : (x1,y1,x2,y2,object_conf, class_score, class_pred)

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Confidence score가 thres 넘는 것만 통과
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]  # 조건을 만족하는 앵커박스만 남긴다
        # image_pred shape (10647, 85)

        # 전부 conf_thres보다 작으면 다음 이미지 수행
        if not image_pred.size(0):
            continue

        # score 계산 (conf * class)
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]  # 클래스 점수가 가장 큰 클래스와 conf를 곱한다 -> score를 만든다

        # 정렬(내림차순 정렬을 하기 위해 score에 -1을 붙인다)
        image_pred = image_pred[
            (-score).argsort()]  # argsort(dim=1)은 행마다 각 열에서 값이 낮은 순으로 인덱스를 저장한다. ,image_pred는 score가 큰 순으로 정렬
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)  # (값, 인덱스) -> 가장 큰 class score과 인덱스를 출력
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)  # 열을 기준으로 합친다.
        #                                                               몇번째 물체인지 나타냄
        # anchor박스가 있는 만큼 nms를 수행
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]

            # 낮은 신뢰도 점수, 큰 IoU 및 일치하는 레이블이있는 상자의 인덱스
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]

            # 신뢰도에 따라 겹치는 bbox 병합
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]  # 최적의 상자 유지
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output  # 출력의 형태 (batch_size, pred_boxes_num, 7)
    # 7 -> x,y,w,h,conf, class_conf, class_pred
    # pred_boxes_num : 각 이미지에 pred_boxes_num개가 있다는 것
