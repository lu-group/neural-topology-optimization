> Under Construction

# neural-topology-optimization for efficient channel design

## Data

The training dataset of neural operators an be downloaded via shared folder [https://drive.google.com/drive/folders/1rOqFQoMn4nR6CvQsGryxRcJPB5d0tgaD?usp=sharing](https://drive.google.com/drive/folders/1rOqFQoMn4nR6CvQsGryxRcJPB5d0tgaD?usp=drive_link).


## Code

1. [Data generation](data_generation): Compile [UDF.c](data_generation/UDF.c) and run [fluent.cas](data_generation/fluent.cas) to generate the dataset.
2. [Neural operators](neural_operator): Run the [train_GC.py](neural_operator/GC/train_GC.py) and [train_GP.py](neural_operator/GP/train_GP.py) to train the neural operators.
3. Run the [main.py](main.py) to perform neural topology optimization.

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:
```
@article{https://doi.org/10.1002/advs.202508386,
author = {Kou, Chenhui and Yin, Yuhui and Zhu, Min and Jia, Shengkun and Luo, Yiqing and Yuan, Xigang and Lu, Lu},
title = {Neural Topology Optimization Via Active Learning for Efficient Channel Design in Turbulent Mass Transfer},
journal = {Advanced Science},
pages  ={e08386},
keywords  ={computational fluid dynamics, active learning, mass transfer enhancement, neural operator, neural topology, topology optimization},
doi  ={https://doi.org/10.1002/advs.202508386},
url  ={https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/advs.202508386},
eprint  ={https://advanced.onlinelibrary.wiley.com/doi/pdf/10.1002/advs.202508386}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
