
# Gradual Domain Adaptation via Gradient Flow

This repository is the official implementation of Gradual Domain Adaptation via Gradient Flow (GGF). 




## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Data Preparation

Download the Portraits dataset from [here](https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=0).

Run `dataset/make_mnist.ipynb` and `dataset/make_portraits.ipynb` to build the UMAP features of source and target domains.



## Training 

First, to train the modules (score network, rectified flow, and initial classifier) in the paper, run this command: 

```train
python train_init.py --task portraits --class_num 2 --save_path save/ --gpu_id 0
python train_init.py --task mnist45 --class_num 10 --save_path save/ --gpu_id 0
python train_init.py --task mnist60 --class_num 10 --save_path save/ --gpu_id 0
```

Second, to gradual generate intermediate domains and update the classifier, run this command: 

```train
python train_ggf.py --task portraits --class_num 2 
python train_ggf.py --task mnist45 --class_num 10 --alpha --iterations --lambda --eta1 --eta2 --eta3 --confidence
python train_ggf.py --task mnist60 --class_num 10 --alpha --iterations --lambda --eta1 --eta2 --eta3 --confidence
```



## Reference

The "Denoise score matching" algorithm is built upon the implementation from https://github.com/Ending2015a/toy_gradlogp.

The codes related to "Rectified Flow" are built upon the tutorial code of the official implementation from https://github.com/gnobitab/RectifiedFlow.



## Citation

```
@inproceedings{
    zhuang2024gradual,
    title={Gradual Domain Adaptation via Gradient Flow},
    author={Zhan Zhuang and Yu Zhang and Ying Wei},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=iTTZFKrlGV}
}
```

