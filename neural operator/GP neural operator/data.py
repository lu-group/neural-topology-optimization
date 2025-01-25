import numpy as np
import pandas as pd
# import torch

# device = torch.device("cuda")
def gen_data(i,j):
    
    filename = "F:/case-pytorch/FNO_Topo/111cvs/v"+str(i)+"N"+str(j)+".csv"

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
    Y = (data[:,11:12]/2000).reshape(1,101,201).astype(np.float32)
    Y = Y+0.26   
    return [X_branch, X_trunk,Y,xx,yy] #,Ve,De,anchors_x 

def dataset(pro):
    X_branch = []
    X_trunk = []
    Y = []
    if pro == 'train':

       for j in [index for index in np.arange(39)]+[index for index in np.linspace(41,101,61,dtype=int)]: 
           for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:          
              X_branch.append([gen_data(i,j)[0]])
              X_trunk.append([gen_data(i,j)[1]])
              Y.append([gen_data(i,j)[2]])
       for j in [index for index in np.linspace(103,112,10,dtype=int)]: 
            for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:          
              X_branch.append([gen_data(i,j)[0]])
              X_trunk.append([gen_data(i,j)[1]])
              Y.append([gen_data(i,j)[2]])
       for j in [2001,2002,2003]: 
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

    
    X_branch = np.concatenate(X_branch) #.astype(np.float32)  
    X_trunk = np.concatenate(X_trunk) #.astype(np.float32)
    Y = np.concatenate(Y) #.astype(np.float32)
    
    return [(X_branch, X_trunk),Y]

