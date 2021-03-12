import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import random

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

def pad_to_square(image, pad_value=0): # 이미지를 정사각형으로 만든다.
    _, h, w = image.shape

    diff = abs(h-w) # h와 w 차이의 절댓값

    if h <= w:
        top = diff//2
        bottom = diff - top
        pad = [0,0,top, bottom] # 한 열에 대해 각각 앞 뒤를 top, bottom 개수만큼 채워준다 ex) (1,1) -> [(0,0), (1,1), (0,0)] (padding 처리)
    else:
        left = diff//2
        right = diff - left
        pad = [left, right, 0, 0] # 한 행에 대해 각각 앞 뒤를 left, right 개수만큼 채워준다 ex) (1,1) --> (0,1,1,0)

    image = F.pad(image, pad, mode='constant', value = pad_value) # image를 패딩해서 정사각형으로 만둔디 반환한다.
    return image, pad

def resize(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)
    # 딥러닝에서 interpolation은 작은 feature의 크기를 크게 변경시킬 때 사용된다.
    # image.unsqueeze(0)는 입력 Tensor
    # size는 interpolate 할 목표 사이즈, 여기서 size는 batch와 channel을 뺀 값이다, (size와 scale factor 중 하나만 입력해야한다)
    # scale factor 또한 interpolate 할 목표 사이즈가 된다.
    # mode는 upsample 하는 방법

class dataset(torch.utils.data.Dataset):
    def __init__(self, list_path:str, image_size : int, augment : bool, multiscale : bool, normalized_labels : True):
        with open(list_path, 'r') as file:
            self.image_files = file.readlines() # 파일의 모든 줄을 읽어서 각각의 줄을 요소로 갖는 리스트로 돌려준다.

        self.label_files = [path.replace('images', 'labels').replace('.png', 'txt').replace('jpg', 'txt').replace('JPECImages', 'labels') for path in self.image_files]
        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels # ?
        self.batch_count = 0

    def __getitem__(self, index): # 슬라이싱을 구현할 수 있도록 도우며 리스트에서 슬라이싱을 하게되면 내부적으로 __getitem__을 실행한다. 따라서 객체에서도 슬라이싱을 하기 위해서는 __getitem__가 필수적이다.
                                  # 이미지를 불러오는 방법이다.
        image_path = self.image_files[index].rstrip()

        # Apply augments
        if self.augment:
            transforms = torchvision.transforms.Compose([  # torchvision.transforms -> 다양한 이미지 변환 기능들을 제공한다. ,torchvision.transforms -> 여러 transform들을 Compose로 구성할 수 있다
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1), # 이미지의 밝기와 대비 및 채도를 임의로 변경한다.
                torchvision.transforms.ToTensor() # 이미지 데이터를 텐서로 바꿔준다.
            ])
        else:
            transforms = torchvision.transforms.ToTensor()   # PIL 이미지 또는 numpy.ndarray를 pytorch의 텐서로 변형시켜 준다.

        # Extract image as Pytorch tensor
        image = transforms(Image.open(image_path).convert('RGB')) # image는 해당 경로에 있는 이미지를 RGB 값으로 변경 시켜서 텐서로 변형된 값이 들어간다

        _, h, w = image.shape
        h_factor, w_factor = (h,w) if self.normalized_labels else (1,1)

        image, pad = pad_to_square(image)
        _, padded_h, padded_w = image.shape

        label_path = self.label_files[index].rstrip()

        targets = None

        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1,5)) # load_path로 부터 텍스트 파일을 한줄 한줄 불러와서 shape를 -1,5로 변환해준 것을 텐서로 만들어 boxes에 저장한다.

            x1 = w_factor * (boxes[:,1] - boxes[:, 3] / 2) # 앵커박스의 좌상단x좌표 * 이미지w크기
            y1 = h_factor * (boxes[:,2] - boxes[:, 4] / 2) # 앵커박스의 좌상단y좌표 * 이미지h크기  , (x1,y1) -> 앵커 박스의 왼쪽 상단 좌표를 의미한다
            x2 = w_factor * (boxes[:,1] + boxes[:, 3] / 2) # 앵커박스의 우하단x좌표 * w크기
            y2 = h_factor * (boxes[:,2] + boxes[:, 4] / 2) # 앵커박스의 우하단y좌표 * h크기, (x2,y2) -> 앵커 박스의 오른쪽 하단 죄표를 의미한다.

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            boxes[:,1] = ((x1+x2)/2)/padded_w   # x
            boxes[:,2] = ((y1+y2)/2)/ padded_h  # y
            boxes[:,3] *= w_factor / padded_w # w
            boxes[:,4] *= h_factor/ padded_h # h

            targets = torch.zeros(len(boxes),6) # 6??
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                image, targets = torch.horizontal_flip(image,targets)

        return image_path, image, targets

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch): # Batch size 만큼 데이터를 불러와서 처리하는 함수
        paths, images, targets = list(zip(*batch)) # ??

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:,0] = i

        try:
            targets = torch.cat(targets,0)
        except RuntimeError:
            targets = None

        # Selects new image size every 10 batches
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(320, 608 + 1, 32))

        # Resize images to input shape
        images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1

        return paths, images, targets






