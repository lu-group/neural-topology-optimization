# neural-topology-optimization

Data

The training dataset of neural operators an be downloaded via shared folder https://drive.google.com/drive/folders/1rOqFQoMn4nR6CvQsGryxRcJPB5d0tgaD?usp=sharing.

[Experiment data](experiment data) is for the original experimental data and calibration curve code.

Code
1. [Data generation](data generation): Compile [udf.c](data_generation/udf.c) and run [fluent.cas](data generation/fluent.cas) to generate the dataset.
2. [Neural operators](neural operators): Run the train_GC.py and train_GP.py to train the neural operators.
3. Run the main.py to perform neural topology optimization.
