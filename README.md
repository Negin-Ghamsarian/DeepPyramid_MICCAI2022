# DeepPyram_MICCAI2022

This repository provides the official PyTorch implementation of DeepPyram (Pyramid View and Deformable Pyramid Reception).

DeepPyram is initially proposed for semantic segmentation in cataract surgery videos, but can be adopted for any medical or general purpose image segmentation problem.

This neural network architecture is especially designed to deal with severe deformations and scale variations by fusing sequential and parallel feature maps adaptively.

**Proposed Pyramid View Fusion and Deformable Pyramid Reception modules:**

<img src="./Figures/PVF-DPR.png" alt="Proposed Pyramid View Fusion and Deformable Pyramid Reception modules." width="500">

---

**Overall architecture of the proposed DeepPyram network:**

<img src="./Figures/BD.png" alt="Overall architecture of the proposed DeepPyram network." width="1200">

**Detailed architecture of the Deformable Pyramid Reception (DPR) and Pyramid
View Fusion (PVF) modules:**

<img src="./Figures/BD-DPR-PVF.png" alt="Detailed architecture of the Deformable Pyramid Reception (DPR) and Pyramid
View Fusion (PVF) modules." width="1200">

**Quantitative comparisons among DeepPyram and rival approaches based on average and standard deviation of IoU.**

<img src="./Figures/IoU_Comparison.png" alt="Quantitative comparisons among DeepPyram and rival approaches based on average and standard deviation of IoU." width="1200">

## Citation
If you use AdaptNet for your research, please cite our paper:

```
@INPROCEEDINGS{DeepPyram,
  author={N. {Ghamsarian} and M. {Taschwer} and R. {Sznitman} and K. {Schoeffmann}},
  booktitle={25th International Conference on Medical Image Computing \& Computer Assisted Interventions (MICCAI 2021)}, 
  title={DeepPyram: Enabling Pyramid View and Deformable Pyramid Reception for Semantic Segmentation in Cataract Surgery Videos}, 
  year={2022},
  volume={},
  number={},
  pages={to appear},}
```

## Acknowledgments

This work was funded by Haag-Streit Switzerland and the FWF Austrian Science Fund under grant P 31486-N31.
