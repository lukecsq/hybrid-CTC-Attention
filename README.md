# hybrid CTC-Attention Decoder

### Dependency
- requirements :Python3, PyTorch 1.1, lmdb, pillow, torchvision



### How to use

First of all,  you should use  *./Tools/genSubword_ch.cpp*  to get statistics on the high-frequency subwords in the dataset.

```shell
$ cd Tools
$ g++ -std=c++11 genSubword_ch.cpp -o genSubword_ch`
$ ./genSubword_ch
```

Then, move the generated file .*/Tools/Chs_subword.txt* to *./config/*

##### Train

```shell
python train.py
```

##### Test

```shell
python test.py
```

