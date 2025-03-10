# Adversarial Representation Learning for Medical Imaging

## ⚠️ WARNING ⚠️ 
**I have archived this repo as it is no longer maintained.**


Brief Description
---
The goal of this work is to investigate unsupervised representation learning methods in the context of medical imaging, with the goal of learning representations useful for the end task of lesion and tumor segmentation.

Deep learning [1, 2] is at the core of recent advances in computer vision. Usually large labeled datasets are used for training in a supervised setting. However, such datasets are not always available. One way to cope with this challenge is to learn representations without labels, and finetune the model for a final task using the labels available [3].

This project focuses on the powerful framework of adversarial learning for learning representations at multiple scales in the medical imaging domain [4, 5, 7].
Specifically, it builds on top of ConSinGAN [7], to generate 2D mammograms based on single samples.

[1] Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep learning (Vol. 1, No. 2). Cambridge: MIT press. https://www.deeplearningbook.org/ <br>
[2] Pouyanfar, Samira, et al. "A survey on deep learning: Algorithms, techniques, and applications." ACM Computing Surveys (CSUR) 51.5 (2018): 1-36. <br>
[3] Brigato, Lorenzo, and Luca Iocchi. "A close look at deep learning with small data." 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021. <br>
[4] Litjens, Geert, et al. "A survey on deep learning in medical image analysis." Medical image analysis 42 (2017): 60-88. <br>
[5] Shaham, Tamar Rott, Tali Dekel, and Tomer Michaeli. "Singan: Learning a generative model from a single natural image." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019. <br>
[6] Kazeminia, Salome, et al. "GANs for medical image analysis." Artificial Intelligence in Medicine (2020): 101938. <br>
[7] Hinz, Tobias, et al. "Improved Techniques for Training Single-Image GANs." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2021). <br>
