# Semantic-Segmentation-of-Recycled-Items
재활용품 분류를 위한 Semantic Segmentation

```
data
    data
      ├── batch_01_vt
      │   ├── 데이터셋 (이미지)
      │   ├── ... 
      │   └── 
      ├── batch_02_vt
      │   ├── 데이터셋 (이미지)
      │   ├── ... 
      │   └── 
      ├── batch_03
      │   ├── 데이터셋 (이미지)
      │   ├── ...
      │   └── 
      ├── train_all.json
      ├── train.json
      ├── val.json
      ├── train_data0.json
      ├── ... 
      ├── valid_data4.json
      ├── train_data_pesudo0.json
      ├── ... 
      ├── valid_data_pesudo4.json
      └── test.json
code
  ├── PyTorch DeepLabv3plus Code.ipynb
  ├── PyTorch EfficientFPN Code.ipynb
  ├── Pesudo Labeling.ipynb
  ├── Ensemble All Models.ipynb
  └── Make A Stratified KFold Json.ipynb
losses
  ├── dice.py
  ├── ...
  └── soft_ce.py
utils.py 

```



## Models 

- DeepLabv3+ plus with Resnext50_32x4d / Resnext101_32x4d
- FPN plus with EfficientNetB4 / EfficientNetB5



## Loss

- CrossEntropyLoss with Label Smoothing 

![](https://github.com/choco9966/Semantic-Segmentation-of-Recycled-Items/blob/main/image/val_loss.PNG?raw=true)



## Metrics 

- mean IoU

![](https://github.com/choco9966/Semantic-Segmentation-of-Recycled-Items/blob/main/image/val_miou.PNG?raw=true)

