# Hybrid CTC-Attention Decoder with Subword Units for the End-to-End Chinese Text Recognition

This is an implementation of paper "Hybrid CTC-Attention Decoder with Subword Units for the End-to-End Chinese Text Recognition". 

## Dependency

- requirements :Python3.5,  PyTorch v1.1.0,  torchvision,  lmdb,  pillow,  numpy

## How to use

Follow the following steps to train a new model on your own dataset.

## Dataset preparation

Download the training, validation and testing dataset

① Variable length Synth-Chs. The dataset can be downloaded from here.

② CASIA-HWDB 2.0-2.2. The dataset can be downloaded from 

[here]: http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

.

③ ICDAR2017 MLT. The dataset can be downloaded from 

[here]: https://rrc.cvc.uab.es/?ch=8&amp;com=downloads

.

### Preparation

1. Create a new LMDB dataset. A python program is provided in ``Tools/create_dataset.py``. Refer to the function ``createDataset`` for details (need to ``pip install lmdb`` first).

2. Create the high-frequency subwords units. A C++ program is provided in ``Tools/genSubword_ch.cpp``. Refer to the function ``splitTxtCh`` for details. 

   ```shell
   $ cd Tools
   $ g++ -std=c++11 genSubword_ch.cpp -o genSubword_ch
   $ ./genSubword_ch
   ```

   Then, move the generated file  ``./Tools/Chs_subword.txt ``  to   ``./config/``

### Training

```shell
python test.py
```

### Testing

```shell
python test.py
```

## Citation

If you find our method useful for your reserach, please cite: