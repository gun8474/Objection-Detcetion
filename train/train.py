import argparse
import os
import time

import torch
import torch.utils.data
from model import Yolov3
from util import utils
from tqdm import trange, tqdm
from util import Dataset


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--dradient_accumulation", type=int, default=1, help = "number of gradient accums before step")
    parser.add_argument("--multiscale_training", type=bool, default=True, help="allow for multi-scale training")
    parser.add_argument("--batch_size", type = int, default=32, help="size of each image batch")
    parser.add_argument("--num_workers", type=int, default=8, help="num of cpu threads to use during batch generation")
    parser.add_argument("--data_config", type=str, default="C:/Users/gun84/PycharmProjects/objectDetection/coco_data.cfg", help="path to data config file")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
    args = parser.parse_args()
    print(args)
    #
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))

    # 데이터셋 설정값 가져오기
    data_config = utils.parse_data_config(args.data_config) #????
    train_path = data_config['train']
    valid_path = data_config['valid']
    num_classes = int(data_config['classes'])
    class_names = utils.load_classes(data_config['names'])


    # 모델 준비
    model = Yolov3.yolov3(args.image_size, num_classes).to(device)
    # model.apply(utils.utils.init_weights_normal)
    # if args.pretrained_weights.endswith('.pth'):
    #     model.load_state_dict(torch.load(args.pretrained_weights))
    # else:
    #     model.load_darknet_weights(args.pretrained_weights)

    # 데이터셋, 데이터 로더 설정
    dataset = Dataset.dataset(train_path, args.image_size, augment=True, multiscale = args.multiscale_training, normalized_labels=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    # learning rate scheduler 설정
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 현재 배치 손실값을 출력하는 tqdm 설정
    loss_log = tqdm(total=0, position=2, bar_format='{desc}', leave=True)

    # Train code
    for epoch in tqdm(range(args.epoch), desc="Epoch"):
        model.train() # 모델을 train mode로 설정

        # 1 epoch의 각 배치에서 처리하는 코드
        for batch_idx, (_,images, targets) in enumerate(tqdm(dataloader, desc='Batch', leave = False)):
            step = len(dataloader) * epoch + batch_idx

            # 이미지와 정답 정보를 GPU로 복사
            images = images.to(device)
            targets = targets.to(device)

            # 순전파(forward), 역전파(backward)
            loss, outputs = model(images, targets)
            loss.backward()

            # Accumulate gradient (기울기 누적)
            if step % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.set_description_str('Loss: {:.6f}'.format(loss.item()))

        # 1개의 epoch 완료 후
        scheduler.step()  # lr_scheduler step 진행
        #checkpoint file 저장
        save_dir = os.path.join('checkpoints', now) # 경로를 병합하여 새 경로 생성
        os.makedirs(save_dir, exist_ok= True)# 새로운 디렉토리 생성
        dataset_name = os.path.split(args.data_config)[-1].split('.')[0] #  디렉토리명과 파일명이 리스트 형태로 나옴
        torch.save(model.state_dict(), os.path.join(save_dir, 'yolov3_{}_{}.pth'.format(dataset_name, epoch)))




