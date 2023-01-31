# 4D_Block_classification_competition

# [Dacon 포디 블럭 구조 추출 AI 경진대회](https://dacon.io/competitions/official/236046/overview/description)

## 개요

포디블록(4Dblock)은 유아기∼전 연령을 대상으로 하는 융합놀이를 위한 조작교구 입니다.

블록놀이는 공간 지각능력과 창의력 발달에 도움이 되는 놀이도구이며, 교육용 블록교구는 다양하게 구성 및 조적하게 되어있어 비구조적인 특징을 갖습니다.

하지만 이러한 비구조적인 특성은 발달특성상 유아들에게는 목적 없는 놀이도구로 소진되기 쉽기 때문에 보다 창의적인 높은 수준의 블록놀이를 촉진하기 위해서 체계적인 쌓기 구조에 대한 사전지식의 지원이 필요합니다.

이를 위해 어린이의 쌓기 구조 데이터를 수집하고 이에 대한 반복적인 블록 쌓기 구조 패턴 인식 및 쌓기 구조의 패턴을 분류하여 효율적이고 유용한 방법 및 해결책을 제시할 수 있을 것입니다.

이 기술은 나아가 오프라인 실험군, 통제군을 대상으로한 공간지각력, 창의성 등 자체 개발 된 평가 툴을 추가 학습시켜 사용자의 융합적 레벨 테스트를 같이 제공하여 블록 놀이&활동을 통한 교육적 진단 서비스로 확장 하고자 합니다.

학습자 선호 유형에 따른 활동 및 프로그램 매칭에 적용할 2D 이미지 기반 블록 구조 추출 AI 모델을 만드는 것이 목표입니다.

---

## EDA

- 모든 클래스 분포 동일(10%)
- train이미지: 배경 X, test이미지: 배경 O
    - train 데이터 셋에 숫자 31600~31711는 배경 이미지 존재
- 레이블 개수 8, 9, 10인 이미지가 다른 레이블 개수의 이미지에 비해 데이터 수가 적음
- train set이 동일한 블럭이 회전한 24개(대부분 24개이고 적게는 한개부터 많게는 50개가 넘어가는 것도 있음)의 이미지들로 구성되어있음
- 초기 데이터 개수 32993
- 모든 블럭이 사진의 가운데에 위치

---

## strategy

validation set 구성: 이미 클래스가 균일한 분포를 가지고 있고 배경합성시 모두 다른 이미지로 합성을 하여 랜덤으로 뽑아도 겹치는 이미지가 없고 train, val 분포 차이가 크지 않을 것이라 생각해 df.sample에 frac을 1, seed를 777로 주고 0.8:0.2(0.1)로 나누고 0.2(0.1)를 validation set으로 사용

### data augmentation

(*validation set 클래스 분포 확인*)

- train이미지: 배경 X, test이미지: 배경 O

→ pixabay에서 indoor, child, tabel로 이미지 합성 시도

| Model | 부가 설명 | val | Public | 실험 |
| --- | --- | --- | --- | --- |
| sample_submission | 모든 predicition이 0 | x | 0.5630136986 | - |
| efficientnet-b0 | baselinecode, 50 epoch | x | 0.7730593607 | 베이스코드에 에폭 수만 증가 |
| convnext_base | 모델 변경, 15 epoch | x | 0.8643835616 | convnext로 모델 변경 |
| convnext_base | background(threshold:170) 합성, 15epoch | x | 0.9141552511 | 블록 이미지에 배경 합성 |

<p align=center>
<img width=404 src="https://user-images.githubusercontent.com/77565951/215736480-34b63faa-89f1-428f-9a2c-1a05dbe31435.png"/>
<img width=404 src="https://user-images.githubusercontent.com/77565951/215736489-28f4b918-d32b-4a94-a8e5-25b6f8b49081.jpeg"/>
<p/>

- 레이블 개수 8, 9, 10인 이미지가 다른 레이블 개수의 이미지에 비해 데이터 수가 적음

→ 기존 블록 이미지를 가져와서 8, 9, 10개인 데이터를 만들어줌(알고리즘 오류로 8, 9, 10은 자체적으로는 중복된 데이터가 소수있었고, 이동 후 블록이 겹치는 경우도 소수 생겼습니다(오른쪽 사진))

→ 31600~31711 배경이 존재하는 것을 나중에 확인하여서 데이터를 수정

<p align=center>
<img width="404" alt="스크린샷 2023-01-20 오후 11 01 41" src="https://user-images.githubusercontent.com/77565951/215736423-9e5dc687-379c-4977-8720-dd1d3f48355e.png">
<img width="404" alt="스크린샷 2023-01-20 오후 11 16 16" src="https://user-images.githubusercontent.com/77565951/215736450-d0dafcf8-35a9-481e-add2-83461f8e15eb.png">
</p>


- test, train에는 대부분 중앙에 이미지가 있었지만 generic한 성능을 위해 block size와 위치를 랜덤으로 재조정하여 이미지를 다시 만듦(8, 9, 10 개수의 레이블 합성할 때도 같은 방식 적용)

블록 레이블을 삭제시켜 정보를 훼손할 수 있는 crop이나 hard augmentation은 사용하지 않음

학습 중 Randomrate90, Blur, RandomBrightnessContrast 적용, resize 384

- 이미지 합성시 노이즈가 있더라도 블록이미지가 사라지는 것을 최대한 억제하기 위해 threshold 170 → 230으로 변경

### hardvoting

- ConvNext L & XL

### label smoothing

- 모든 모델에 대해 I, J, G가 낮게 나타나는 것을 확인, Class 분포가 같은데도 이와 같은 현상은 Difficulty문제로 해석해 CDB loss를 적용해 (0.9543→0.9552)0.001정도의 미미한 향상이 있었습니다.
<p align=center>
<img width="430" alt="스크린샷 2023-01-31 오후 7 12 02" src="https://user-images.githubusercontent.com/77565951/215736462-22703393-4257-47d3-b5e8-0923e68126b7.png">
</p>
    

### I, J, H oversampling

위의 성능이 낮게 나오는 결과 때문에 배경을 다르게 하여 학습 시켰지만 성능의 향상은 없었음

### Model

Convnext 사용

classifier에 mlp계층을 더 추가해 사용

## BaseLine 모델

| Model | 부가 설명 | val | Public | 실험 |
| --- | --- | --- | --- | --- |
| sample_submission | 모든 predicition이 0 |  | 0.5630136986 | - |
| efficientnet-b0 | baselinecode, 50 epoch |  | 0.7730593607 | 베이스코드에 에폭 수만 증가 |
| convnext_base | 모델 변경, 15 epoch |  | 0.8643835616 | convnext로 모델 변경베이스라인 코드 대비 성능 증가 |
| swin_t_base | 모델 변경, 15epoch |  | 0.8121004566 | swin_t로 모델 변경 convnext 대비 성능 감소 |
| convnext_base | background(170) 합성, 15epoch |  | 0.9141552511 | test이미지에 맞게 train이미지에 배경 합성, 성능 증가 |
| convnext_base | backgorund(170) 합성, 50epoch |  | 0.9200913242 | 이전 대비 에폭 증가, 성능 증가 |
| convnext_base | background(220) 합성, 50epoch |  | 0.9303652968 | background 합성 시에 블럭 이미지를 더 보존(170→220), 성능 증가 |
| convnext_large | background(220) 합성, 15epoch |  | 0.9292237443 | large로 모델 변경, base대비 15에폭에서는 성능 증가 |
| convnext_large | background(220) 합성, 50epoch |  | 0.9257990868 | 이전 대비 에폭 증가, 성능 감소 (과적합) |
| convnext_tiny | 파인튜닝 cifar10, 220합성, 20 epoch  |  | 0.9162100457 | cifar10 파인튜닝 된 convenxt tiny사용, 학습 속도는 이전 모델들 대비 상승, 성능은 감소 |
| coatnet_small | background(220) 합성, 10epoch |  | 0.9141552511 | 모델 변경, 성능 감소 |
| maxvit_large | background(220) 합성, 15epoch |  | 0.901369863 | 모델 변경, 성능 감소 |
| convnext_xlarge | background(230) 합성, 10epoch |  | 0.9340182648 | xlarge모델로 변경, 이미지 합성 시 블럭이미지 보존 정도(220→230) 증가, 성능 증가 |
| convnext_xlarge | background(230) 합성, 20epoch |  | 0.9404109589 | 이전 대비 에폭 수 증가, 성능 증가 |
| convnext_xlarge | background(230) 합성, 50epoch(early stop 38), 모든 배경 이미지 다르게 합성+, block image random resize, random move, 8, 9, 10 레이블 합성시에도 동일하게 적용+레이블 합성시 같은 이미지 최대 2개까지만 쓰도록 변경 |  | 0.95456 | 에폭증가, 성능증가 |
| convnext_xlarge | 위와 동일, classifier 2 layer mlp |  | 0.957 | 좀더 generic 한 성능을 가지는 듯 하다 |

---
### 아쉬운 점
Q2L 적용 못해봤음

- 배경합성시 문제점
    - 크롤링시 사이트에서 알파벳 순으로 정렬되어있고, 블럭 이미지도 정렬이 되어 있는 상황에서 섞어서 합성한 것이 아닌 있는 순서 그대로 합성을 진행해 분포가 다양해지지 못한 문제가 있음(즉, 한 레이블 이미지에 같은 키워드의 이미지만 합성되었을 가능성이 높아졌다라고 볼 수 있습니다)

i, j, g 만 따로 추론하는 모델을 만들어서 정답을 합쳐서 제출 했지만 성능이 좋아지지 않았음(같은 모델, 같은 데이터로만 실험해보았음)

validation set 평가와 public test datset의 평가가 일치하지 못한 문제가 있었다.(학습할 수록 validation이 계속 올라갔음: overfitting으로 인한 성능 하락)
- 합성한 배경을 모두 다르게 해서 train과 val 데이터와 겹치지 않을 것이라 생각했지만 overfitting을 잡아내지 못해 train과 validation의 이미지가 겹치는 문제가 있었다고 생각됨
