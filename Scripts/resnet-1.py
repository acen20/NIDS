#!/usr/bin/env python
# coding: utf-8


import os
exp_num = 1 #resnet18
os.environ["WANDB_API_KEY"] = "00c5bcfd2b2fbe9bce38152923c98635448f8c6f"
EXPERIMENT_NAME = f'ResNET18_{exp_num}'
NOTE = "resnet18"


# In[ ]:


os.system('mkdir ../dataset') 
os.system('mkdir ../models')
os.system('wget -nc -q -O ../dataset/KDDTrain+.txt https://raw.githubusercontent.com/acen20/DEL/master/deep-ensemble-jet/dataset/KDDTrain%2B.txt')
os.system('wget -nc -q -O ../dataset/KDDTest+.txt https://raw.githubusercontent.com/acen20/DEL/master/deep-ensemble-jet/dataset/KDDTest%2B.txt')
os.system('wget -nc -q -O ../dataset/KDDTrain+_20Percent.txt https://raw.githubusercontent.com/acen20/DEL/master/deep-ensemble-jet/dataset/KDDTrain%2B_20Percent.txt')
os.system('wget -nc -q -O ../dataset/KDDTest-21.txt https://raw.githubusercontent.com/acen20/DEL/master/deep-ensemble-jet/dataset/KDDTest-21.txt')


# In[ ]:


os.system('pip install wandb -q')
os.system(f'wandb login {os.environ["WANDB_API_KEY"]}')
import wandb


# In[ ]:


import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
#plt.style.use('grayscale')
from sklearn.metrics import precision_recall_curve,RocCurveDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import time
import pickle


# In[ ]:


resnet = resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False


# In[ ]:


class CLF(nn.Module):
    def __init__(self):
        super(CLF, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.clf(x)


# In[ ]:


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


print(f'Using {device} as device')


# In[ ]:


clf = CLF()
model = nn.Sequential(resnet, clf).to(device)


# In[ ]:


def normalize_data(X):
  mms = MinMaxScaler()
  return mms.fit_transform(X)


# In[ ]:


def visualize_train_loss(loss_1, labels, epochs, loss_2 = None):
  plt.figure(figsize=(10,4))
  plt.plot(loss_1, linewidth=2)
  if loss_2:
    plt.plot(loss_2, linewidth=2)
  plt.legend(labels)
  plt.ylabel("loss")
  _ = plt.xlabel(f"epochs ({epochs})")


# In[ ]:


def get_cm(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    tn=cm[1][1] #tn
    fn=cm[0][1] #fn
    fp=cm[1][0] #fp
    tp=cm[0][0] #tp
    acc= (tp+tn)/(tp+tn+fn+fp)
    epsilon = 1e-7 # is used so that to avoid divide by zero error
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    sensitivity,specificity = tp/(tp+fn),tn/(tn+fp)
    print("Test accuracy is:"+str(format(acc,'.4f')))
    print("Precision: "+str(format(precision,'.4f'))+"\nRecall: "+str(format(recall,'.4f')))
    return cm


# In[ ]:


def disp_PR(y_test, probs, preds, cm, title):
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, probs)
    lr_f1, lr_auc = f1_score(y_test, preds), auc(lr_recall, lr_precision)
    no_skill = len(y_test[y_test==0]) / len(y_test)
    acc = accuracy_score(y_test, preds)
    plt.plot(lr_recall, lr_precision, marker='.', alpha=0.5)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(title)

    # show the plot
    plt.show()
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['normal', 'malicious'])
    disp.plot(cmap="Blues", colorbar = False)
    print(f"F1:\t {lr_f1*100:.3f}%")
    print(f"AUC:\t {lr_auc*100:.3f}%")
    print(f"Accuracy:{acc*100:.3f}%")


# ## Preprocessing

# In[ ]:


df=pd.read_csv('../dataset/KDDTrain+.txt', header=None)
df_20p=pd.read_csv('../dataset/KDDTrain+_20Percent.txt', header=None)
df2=pd.read_csv('../dataset/KDDTest+.txt', header=None)
df3=pd.read_csv('../dataset/KDDTest-21.txt', header=None)


# In[ ]:


df.head()


# In[ ]:


df.columns = [str(i) for i in range(0, len(df.columns))]
df2.columns = [str(i) for i in range(0, len(df2.columns))]
df3.columns = [str(i) for i in range(0, len(df3.columns))]


# In[ ]:


# Replacing Null values with 0. ML classifer cannot learn on Null values
df.fillna(0, inplace=True)
df2.fillna(0, inplace=True)
df3.fillna(0, inplace=True)


# In[ ]:


# Peak on dataset
df.head()


# In[ ]:


df.shape


# In[ ]:


# Peak on the dataset
df.drop('42',axis=1, inplace=True)
df2.drop('42',axis=1, inplace=True)
df3.drop('42',axis=1, inplace=True)
df.head()


# In[ ]:


sub_classes = df.iloc[:,41].value_counts()


# In[ ]:


print(sub_classes)


# In[ ]:


r2l = ['ftp_write','guess_passwd', 'imap', 'multihop', 'phf', 'spy','warezclient','warezmaster','xlock', 'xsnoop','named',
       'sendmail','snmpgetattack', 'snmpguess','httptunnel']
u2r = ['buffer_overflow', 'loadmodule','perl','ps','rootkit','sqlattack','xterm']
dos = ['back', 'land', 'neptune', 'smurf', 'teardrop','pod','mailbomb', 'processtable','udpstorm', 'worm','apache2']
probe = ['ipsweep', 'nmap', 'portsweep', 'satan','saint','mscan']
normal = ['normal']



# In[ ]:


df_X = df.drop('41', axis=1)

print(df_20p[1])

le = LabelEncoder()
for i in df_X:
  if df_X[i].dtype=='object':
     le.fit(df_X[i])
     df_X[i] = le.transform(df_X[i])
     df2[i] = le.transform(df2[i])
     df3[i] = le.transform(df3[i])
     df_20p[i] = le.transform(df_20p[i])

df.iloc[:,:40] = df_X


# In[ ]:


novel_attacks = df2.drop(df2[df2['41'].isin(df['41'])].index)


# In[ ]:


novel_attacks = novel_attacks.drop('41', axis=1)


# In[ ]:


df.iloc[:,:40] = normalize_data(df.iloc[:,:40])
df2.iloc[:,:40] = normalize_data(df2.iloc[:,:40])
df3.iloc[:,:40] = normalize_data(df3.iloc[:,:40])
df_20p.iloc[:,:40] = normalize_data(df_20p.iloc[:,:40])
novel_attacks = normalize_data(novel_attacks)


# In[ ]:


u2r_attacks = df[df['41'].apply(lambda x: x in u2r)].copy()
r2l_attacks = df[df['41'].apply(lambda x: x in r2l)].copy()
dos_attacks = df[df['41'].apply(lambda x: x in dos)].copy()
probe_attacks = df[df['41'].apply(lambda x: x in probe)].copy()
normal_traffic = df[df['41'].apply(lambda x: x in normal)].copy()

df['41'] = df['41'].map(
    lambda x: 'malicious' if x in r2l 
    else 'malicious' if x in u2r 
    else 'malicious' if x in dos 
    else 'malicious' if x in probe 
    else 'normal' if x is 'normal'
    else x
    )

df2['41'] = df2['41'].map(
    lambda x: 'malicious' if x in r2l 
    else 'malicious' if x in u2r 
    else 'malicious' if x in dos 
    else 'malicious' if x in probe 
    else 'normal' if x is 'normal'
    else x
    )

df3['41'] = df3['41'].map(
    lambda x: 'malicious' if x in r2l 
    else 'malicious' if x in u2r 
    else 'malicious' if x in dos 
    else 'malicious' if x in probe 
    else 'normal' if x is 'normal'
    else x
    )

df_20p['41'] = df_20p['41'].map(
    lambda x: 'malicious' if x in r2l 
    else 'malicious' if x in u2r 
    else 'malicious' if x in dos 
    else 'malicious' if x in probe 
    else 'normal' if x is 'normal'
    else x
    )


# ### Training set value counts

# In[ ]:


df['41'].value_counts()


# ### Testing set value counts

# In[ ]:


print(df2['41'].value_counts())


# In[ ]:


#In case of multi-class classification
#df_Y = le.fit(df['41']).transform(df['41'])
#df.iloc[:,41] = df_Y
#df_Y = le.transform(df2['41'])
#df2.iloc[:,41] = df_Y

#In case of binary classification
df.iloc[:,41] = df['41'].map(lambda x: 0 if x=='normal' else 1)
df2.iloc[:,41] = df2['41'].map(lambda x: 0 if x=='normal' else 1)
df3.iloc[:,41] = df3['41'].map(lambda x: 0 if x=='normal' else 1)
df_20p.iloc[:,41] = df_20p['41'].map(lambda x: 0 if x=='normal' else 1)


#X_train = df.drop(['41'],axis=1)
#y_train = df['41']

X_train = df_20p.drop(['41'],axis=1)
y_train = df_20p['41']

X_test = df2.drop(['41'],axis=1)
y_test = df2['41']
X_test = torch.tensor(np.array(X_test), dtype=torch.float, device=device)

X_test21 = df3.drop(['41'],axis=1)
X_test21 = torch.tensor(np.array(X_test21), dtype=torch.float, device=device)
y_test21 = torch.tensor(df3['41'].to_numpy(), dtype=torch.float)


# **Creating TensorDataset**

# In[ ]:


pt_x_train = torch.tensor(X_train.to_numpy(), dtype=torch.float).to(device)
y = nn.functional.one_hot(torch.tensor(y_train.to_numpy(dtype='int')))
pt_y_train = y.float().to(device)
novel_attacks = torch.tensor(novel_attacks, dtype=torch.float).to(device)
tensor_dataset = TensorDataset(pt_x_train, pt_y_train)


# In[ ]:


extras = [[0,0,0,0,0,0,0,0] for _ in range(len(df))]


# In[ ]:


pt_x_train.shape, pt_y_train.shape


# ## **MLP**

# In[ ]:


class MLP(nn.Module):
  data_dim = 41
  def __init__(self, hidden_size):
      super(MLP, self).__init__()
      self.mlp = nn.Sequential(
          nn.Linear(self.data_dim, hidden_size),
          nn.ReLU(),
          nn.BatchNorm1d(hidden_size),
          nn.Linear(hidden_size, hidden_size),
          nn.ReLU(),
          nn.BatchNorm1d(hidden_size)
      )

      self.clf = nn.Sequential(
          nn.Linear(hidden_size,2),
          nn.Sigmoid()
      )

  def forward(self, x):
      features = self.mlp(x)
      return self.clf(features)


# In[ ]:


def validation_accuracy(model, X, y):
  model.eval()
  with torch.no_grad():
    lr_probs = model(transform_input(X)).detach().squeeze()
  preds = torch.argmax(lr_probs, dim=1).cpu()
  acc = accuracy_score(y, preds)
  model.train()
  return acc


# In[ ]:


hidden_sizes = [80]
batch_sizes = [64]
epochs_ = [50]
learning_rates = [1e-2]


# In[ ]:


MAX_COMBINATIONS = 1


# In[ ]:


hyper_space = []
for hidden_size in hidden_sizes:
    for epochs in epochs_:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                hyper_space.append((hidden_size, epochs, lr, batch_size))


# In[ ]:


def transform_input(X):
    extras = [[0,0,0,0,0,0,0,0] for _ in range(X.shape[0])]
    extras = torch.tensor(extras, dtype=torch.float).to(device)
    X = torch.cat((X, extras), dim=1)
    X = X.reshape(-1,7,7)
    upsample = nn.Upsample(scale_factor=32)
    X = upsample(X.view(-1,1,7,7))
    X = X.expand(-1,3,-1,-1)
    return X


# In[ ]:


start_time = time.time()


# In[ ]:


grid_scores = []
iters = 1
#Creating a Random Search
for _ in range(MAX_COMBINATIONS):
    hidden_size, epochs, lr, batch_size = hyper_space[np.random.randint(0, len(hyper_space))]
    #hyper_space.remove((hidden_size, epochs, lr, batch_size))
    #Model init
    pt_train = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    #model = model().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    #Training Loop
    run = wandb.init(entity='ahsen', project='nids', 
                     name=f'{EXPERIMENT_NAME}_{iters}', 
                     group=EXPERIMENT_NAME, reinit=True,
                     notes=NOTE
                     )
    
    wandb.config["lr"] = lr
    wandb.config["batch_size"] = batch_size
    wandb.config["epochs"] = epochs
    wandb.config["hidden_size"] = hidden_size

    print('========================================================')
    print(f"Hidden Size:{hidden_size}\tEpochs:{epochs}\tLR:{lr}\tBatch Size:{batch_size}")
    num_epochs = epochs
    losses = []
    for epoch in range(num_epochs):
        for instance, y in pt_train:
            
            instance = transform_input(instance)
            
            output = model(instance)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())
        acc = validation_accuracy(model, X_test, y_test)
        acc_21 = validation_accuracy(model, X_test21, y_test21)
        novel_acc = validation_accuracy(model, novel_attacks, 
                                        torch.ones((novel_attacks.shape[0],)))
        wandb.log({'loss':loss.item()})
        print('------------------------------------------------------------------------------')
        print(f'Epoch:\t\t{epoch+1}/{epochs}\nLoss:\t\t{loss.item():.4f}\nAcc:\t\t{acc:.4f}\nAcc21:\t\t{acc_21:.4f}\nNovel:\t\t{novel_acc:.4f}')

    print("*************************************************")
    print(f'Final Score for (H:{hidden_size}, Ep:{epochs}, LR:{lr}, B:{batch_size})')
    print(f'Acc:\t\t{acc:.4f}')
    print(f'Acc21:\t\t{acc_21:.4f}')
    print(f'Novel:\t\t{novel_acc:.4f}')
    #Save scores
    obj = {
      "name":f'{EXPERIMENT_NAME}_{iters}',
      "config":{
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "acc": acc,
        "acc_21": acc_21,
        "novel_acc": novel_acc,
        "hidden_size": hidden_size
        },
      "weights": model.state_dict(),
      "losses": losses
    }
    wandb.log({"acc":acc, "acc_21":acc_21, "novel_acc":novel_acc})
    run.finish()
    grid_scores.append(obj)
    elapsed = time.time() - start_time
    print(f'Time Elapsed:\t{elapsed:.0f} seconds')
    iters = iters + 1
    with open('model.config', 'wb') as config_file:
        pickle.dump(obj, config_file)


# In[ ]:


highest = 0
for i,score in enumerate(grid_scores):
  if score['config']['acc'] > grid_scores[highest]['config']['acc']:
    highest = i


# In[ ]:


end_time = time.time()
print("=====================================")
print(f'Total time taken: {int(end_time-start_time)} seconds')
print(f'Best scores with:\t{grid_scores[highest]["name"]}')
print(f'{grid_scores[highest]["config"]}')


# In[ ]:


losses = grid_scores[highest]['losses']
num_epochs = grid_scores[highest]['config']['epochs']
plt.plot(losses)
plt.xlabel(f"epochs({num_epochs})")
plt.ylabel(f"loss")
_ = plt.legend(['Loss'])


# In[ ]:


mlp = MLP(grid_scores[highest]['config']['hidden_size']).to(device)
mlp.load_state_dict(grid_scores[highest]['weights'])
mlp.eval()
with torch.no_grad():
    lr_probs = mlp(X_test).detach().squeeze()
    lr_probs_novel = mlp(novel_attacks).detach().squeeze()
    lr_probs_21 = mlp(X_test21).detach().squeeze()

probs = torch.max(lr_probs, dim=1)
probs_novel = torch.max(lr_probs_novel, dim=1)
probs_21 = torch.max(lr_probs_21, dim=1)

idxs, scores = probs.indices, probs.values
probs = [scores[i].item() if idxs[i]==1 else 1-scores[i].item() for i in range(len(idxs))]

idxs, scores = probs_novel.indices, probs_novel.values
probs_novel = [scores[i].item() if idxs[i]==1 else 1-scores[i].item() for i in range(len(idxs))]

idxs, scores = probs_21.indices, probs_21.values
probs_21 = [scores[i].item() if idxs[i]==1 else 1-scores[i].item() for i in range(len(idxs))]

preds = torch.argmax(lr_probs, dim=1).cpu()
preds_novel = torch.argmax(lr_probs_novel, dim=1).cpu()
preds_21 = torch.argmax(lr_probs_21, dim=1).cpu()

lr_probs = torch.max(lr_probs, dim=1).values
lr_probs_novel = torch.max(lr_probs_novel, dim=1).values
lr_probs_21 = torch.max(lr_probs_21, dim=1).values

probs = np.array(probs)
probs_novel = np.array(probs_novel)
probs_21 = np.array(probs_21)


# In[ ]:


suspicious = probs[(probs>0.4) & (probs<0.5)].shape[0]


# In[ ]:


print(f'{suspicious} packets are suspicious')


# In[ ]:


cm = get_cm(y_test, preds)


# In[ ]:


cm_21 = get_cm(y_test21, preds_21)


# In[ ]:


cm_novel = get_cm(np.ones(novel_attacks.shape[0],), preds_novel)


# In[ ]:


disp_PR(y_test, probs, preds, cm, title = '')


# In[ ]:


disp_PR(y_test21, probs_21, preds_21, cm_21, title = 'Test21')


# In[ ]:


_ = RocCurveDisplay.from_predictions(y_test,probs)


# In[ ]:


_ = RocCurveDisplay.from_predictions(y_test21,probs_21)


# In[ ]:


wandb.finish()


# In[ ]:


torch.save(mlp.state_dict(), 'mlp.pt')

