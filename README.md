# Caffe2Pytorch

### Introduction

This is a tool for changing Caffe model to Pytorch model. I borrow the main framework from [xiaohang's CaffeNet](https://github.com/marvis/pytorch-caffe). I modify the structure and add more supports to them.

Given a `.prototxt` and a `.caffemodel`, the conversion code generates a `.pth`. You can use the Pytorch model by the `.prototxt` and the `.pth`.

There will be lots of layers in Caffe, it is impossible to know how all layers are constructed. Now the code supports these layers:

```python
'Data', 'AnnotatedData', 'Pooling', 'Eltwise', 'ReLU', 'PReLU', 'Permute', 'Flatten', 'Slice', 'Concat', 'Softmax', 'SoftmaxWithLoss', 'LRN', 'Dropout', 'Reshape', 'PriorBox', 'DetectionOutput'
```

### Dependency

**General requirement:**

`python2` or `python3` are both OK, depend on your `pycaffe` API.

`pytorch` >= 0.4

**Special requirement:**

Only the conversion code requires `pycaffe`.

### Usage

**Conversion**

```
python caffe2pth_convertor.py \
 --prototxt=YOUT_PROTOTXT_PATH \
 --caffemodel=YOUT_CAFFEMODEL_PATH \
 --pthmodel=OUTPUT_PTHMODEL_PATH
```

**Use the model in Pytorch**

```python
from caffe2pth.caffenet import *

net = CaffeNet(YOUT_PROTOTXT_PATH)
net.load_state_dict(torch.load(OUTPUT_PTHMODEL_PATH))
```

