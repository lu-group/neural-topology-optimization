1. Generating dataset for training and testing neural operators shared in Section 'Data availability'.
2. Use the code in the 'GP neural operator' and 'GC neural operator' folders to sequentially train the GP and GC operators:
     Build the training and test dataset using 'data.py' 
     Train the neural operator using 'train.py'.
      If necessary, use `model.py` to modify the parameters of the neural operator.
3. Save the trained neural operators and use the `main` file to load them to implement the optimization algorithm.