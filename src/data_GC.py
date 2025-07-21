import numpy as np
import pandas as pd
# import torch

# device = torch.device("cuda")


def gen_data(i,j):
    ### i means index of inlet velocities; j means index of different topology structure   
    filename = "data/v"+str(i)+"N"+str(j)+".csv"

    df=pd.DataFrame(pd.read_csv(filename))
    df.columns = df.columns.str.replace(' ', '')    
    df.columns = pd.to_numeric(df.columns, errors='ignore')
    df = df.round({'x-coordinate':4,'y-coordinate':4})
    df.sort_values(by=['y-coordinate','x-coordinate'], inplace = True)
    
    data = np.array(df)
    xx = data[:,1:2].reshape(101,201).astype(np.float32)
    yy = data[:,2:3].reshape(101,201).astype(np.float32)
    X_branch=data[:,3:4].reshape(1,101,201)[:,:,60:181].astype(np.float32) 
    X_trunk=np.array([i]).astype(np.float32)
    Y = data[:,5:6].reshape(1,101,201).astype(np.float32)
        
    return [X_branch, X_trunk,Y,xx,yy] #,Ve,De,anchors_x 

def dataset(pro):
    X_branch = []
    X_trunk = []
    Y = []
    if pro == 'train':
     ### Initial dataset
      for j in [index for index in np.arange(39)]+[99]: 
         for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:      
              X_branch.append([gen_data(i,j)[0]])
              X_trunk.append([gen_data(i,j)[1]])
              Y.append([gen_data(i,j)[2]])
      ### 2nd actively adding data
      for j in [1001,1002,1003]: 
          for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:          
               X_branch.append([gen_data(i,j)[0]])
               X_trunk.append([gen_data(i,j)[1]])
               Y.append([gen_data(i,j)[2]])
      
              
    if pro == 'test':
      for j in [39,40]:        
         for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
              X_branch.append([gen_data(i,j)[0]])
              X_trunk.append([gen_data(i,j)[1]])
              Y.append([gen_data(i,j)[2]]) 
    # else:
    #   raise NotImplementedError("task name should be 'train' or 'test'") 
    
    X_branch = np.concatenate(X_branch) #.astype(np.float32)  
    X_trunk = np.concatenate(X_trunk) #.astype(np.float32)
    Y = np.concatenate(Y) #.astype(np.float32)
    
    return [(X_branch, X_trunk),Y]

