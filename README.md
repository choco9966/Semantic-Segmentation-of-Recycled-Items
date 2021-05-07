# Semantic-Segmentation-of-Recycled-Items
재활용품 분류를 위한 Semantic Segmentation

## Dataset 경로 

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

## TTA 

- Vertical Flip + Multi Scale 

```python
import ttach as tta

transforms = tta.Compose(
    [
        tta.VerticalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25])
    ]
)
```



## Post Processing 

- CRF 

```Python
h, w = 512, 512
def dense_crf(probs, img=None, n_classes=12, n_iters=10, scale_factor=1):
    c,h,w = probs.shape
    
    if img is not None:
        assert(img.shape[1:3] == (h, w))
        img = np.transpose(img,(1,2,0)).copy(order='C')
        img = np.uint8(255 * img)

    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(3,3), compat=10)
    d.addPairwiseBilateral(sxy=10, srgb=5, rgbim=np.copy(img), compat=10)
    Q = d.inference(n_iters)

    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w))
    return preds
```

