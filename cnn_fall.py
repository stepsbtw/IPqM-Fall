import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset

torch.backends.cudnn.benchmark=True

DATASET_DIR=Path("IPqM-Fall/windowed_synchronized")

CHECKPOINT_DIR=Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHEST_DIR=CHECKPOINT_DIR/"CHEST"
LEFT_DIR=CHECKPOINT_DIR/"LEFT"
RIGHT_DIR=CHECKPOINT_DIR/"RIGHT"

CHEST_DIR.mkdir(exist_ok=True)
LEFT_DIR.mkdir(exist_ok=True)
RIGHT_DIR.mkdir(exist_ok=True)

RESULTS_DIR=Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE=RESULTS_DIR/"results.json"

FEATURE_COLUMNS=["ax","ay","az","amag","wx","wy","wz","wmag"]

WINDOW_SAMPLES=180

BATCH_SIZE=256
EPOCHS=80
LEARNING_RATE=3e-4
DROPOUT=0.35
NUM_CLASSES=2
EARLY_STOPPING_PATIENCE=10

NUM_WORKERS=2
PIN_MEMORY=True

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleSensorDataset(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X.transpose(0,2,1),dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

class CNN1Conv(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(num_features,64,kernel_size=4,padding=2),nn.ReLU(),nn.MaxPool1d(kernel_size=3),nn.Dropout(DROPOUT))
        conv_out_length=WINDOW_SAMPLES+2*2-4+1
        pool_out_length=conv_out_length//3
        flattened_size=64*pool_out_length
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(flattened_size,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,NUM_CLASSES))
    def forward(self,x):
        return self.classifier(self.features(x))

def train_epoch(model,loader,criterion,optimizer):
    model.train()
    total_loss=0
    for X_batch,y_batch in loader:
        X_batch=X_batch.to(DEVICE,non_blocking=True)
        y_batch=y_batch.to(DEVICE,non_blocking=True)
        optimizer.zero_grad()
        outputs=model(X_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(loader)

def predict(model,loader):
    model.eval()
    y_true=[]
    y_pred=[]
    y_prob=[]
    with torch.no_grad():
        for X_batch,y_batch in loader:
            X_batch=X_batch.to(DEVICE,non_blocking=True)
            outputs=model(X_batch)
            probs=F.softmax(outputs,dim=1)
            preds=torch.argmax(probs,dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:,1].cpu().numpy())
    return np.array(y_true),np.array(y_pred),np.array(y_prob)

def compute_metrics(y_true,y_pred):
    acc=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred,pos_label=1,zero_division=0)
    recall=recall_score(y_true,y_pred,pos_label=1,zero_division=0)
    f1=f1_score(y_true,y_pred,pos_label=1,zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {"accuracy":float(acc),"precision":float(precision),"recall":float(recall),"f1":float(f1),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn)}

def train_single_sensor_model(sensor_name,X_train,X_test,y_train,y_test,test_subject,fold_number,save_dir):

    ckpt_path=save_dir/f"{sensor_name}_fold_{fold_number}_subject_{test_subject}.pth"
    mean_path=save_dir/f"{sensor_name}_fold_{fold_number}_mean.npy"
    std_path=save_dir/f"{sensor_name}_fold_{fold_number}_std.npy"

    train_mean=X_train.mean(axis=(0,1),keepdims=True)
    train_std=X_train.std(axis=(0,1),keepdims=True)+1e-8

    np.save(mean_path,train_mean)
    np.save(std_path,train_std)

    X_train=(X_train-train_mean)/train_std
    X_test=(X_test-train_mean)/train_std

    train_dataset=SingleSensorDataset(X_train,y_train)
    test_dataset=SingleSensorDataset(X_test,y_test)

    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,persistent_workers=NUM_WORKERS>0)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,persistent_workers=NUM_WORKERS>0)

    model=CNN1Conv(len(FEATURE_COLUMNS)).to(DEVICE)

    if torch.cuda.device_count()>1:model=nn.DataParallel(model)

    fall_count=np.sum(y_train==1)
    nonfall_count=np.sum(y_train==0)

    class_weights=torch.tensor([1.0/nonfall_count,1.0/fall_count],dtype=torch.float32).to(DEVICE)

    criterion=nn.CrossEntropyLoss(weight=class_weights)

    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    if ckpt_path.exists():

        model.load_state_dict(torch.load(ckpt_path,map_location=DEVICE))

    else:

        best_loss=float("inf")
        patience_counter=0

        epoch_pbar=tqdm(range(EPOCHS),desc=f"{sensor_name} Fold {fold_number}",leave=False)

        for epoch in epoch_pbar:

            loss=train_epoch(model,train_loader,criterion,optimizer)

            epoch_pbar.set_postfix(loss=f"{loss:.4f}",best=f"{best_loss:.4f}")

            if loss<best_loss:

                best_loss=loss
                patience_counter=0

                torch.save(model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),ckpt_path)

            else:

                patience_counter+=1

            if patience_counter>=EARLY_STOPPING_PATIENCE:
                break

        model.load_state_dict(torch.load(ckpt_path,map_location=DEVICE))

    y_true,y_pred,y_prob=predict(model,test_loader)

    metrics=compute_metrics(y_true,y_pred)

    metrics["fold"]=fold_number
    metrics["test_subject"]=str(test_subject)
    metrics["checkpoint"]=str(ckpt_path)
    metrics["mean_path"]=str(mean_path)
    metrics["std_path"]=str(std_path)

    return {"model":model,"metrics":metrics,"y_true":y_true,"y_pred":y_pred,"y_prob":y_prob}

def train_final_model(sensor_name,X,y,save_dir):

    final_ckpt=save_dir/f"{sensor_name}_FINAL.pth"
    final_mean_path=save_dir/f"{sensor_name}_FINAL_mean.npy"
    final_std_path=save_dir/f"{sensor_name}_FINAL_std.npy"

    if final_ckpt.exists():
        return

    final_mean=X.mean(axis=(0,1),keepdims=True)
    final_std=X.std(axis=(0,1),keepdims=True)+1e-8

    np.save(final_mean_path,final_mean)
    np.save(final_std_path,final_std)

    X=(X-final_mean)/final_std

    dataset=TensorDataset(torch.tensor(X.transpose(0,2,1),dtype=torch.float32),torch.tensor(y,dtype=torch.long))

    loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,persistent_workers=NUM_WORKERS>0)

    model=CNN1Conv(len(FEATURE_COLUMNS)).to(DEVICE)

    if torch.cuda.device_count()>1:model=nn.DataParallel(model)

    fall_count=np.sum(y==1)
    nonfall_count=np.sum(y==0)

    class_weights=torch.tensor([1.0/nonfall_count,1.0/fall_count],dtype=torch.float32).to(DEVICE)

    criterion=nn.CrossEntropyLoss(weight=class_weights)

    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    best_loss=float("inf")
    patience_counter=0

    epoch_pbar=tqdm(range(EPOCHS),desc=f"{sensor_name} FINAL",leave=False)

    for epoch in epoch_pbar:

        loss=train_epoch(model,loader,criterion,optimizer)

        epoch_pbar.set_postfix(loss=f"{loss:.4f}",best=f"{best_loss:.4f}")

        if loss<best_loss:

            best_loss=loss
            patience_counter=0

            torch.save(model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),final_ckpt)

        else:

            patience_counter+=1

        if patience_counter>=EARLY_STOPPING_PATIENCE:
            break

if __name__=="__main__":

    print(f"DEVICE: {DEVICE}")

    X_chest=np.load(DATASET_DIR/"X_chest.npy")
    X_left=np.load(DATASET_DIR/"X_left.npy")
    X_right=np.load(DATASET_DIR/"X_right.npy")

    y=np.load(DATASET_DIR/"y.npy")
    groups=np.load(DATASET_DIR/"groups.npy")

    print(f"CHEST: {X_chest.shape}")
    print(f"LEFT: {X_left.shape}")
    print(f"RIGHT: {X_right.shape}")

    logo=LeaveOneGroupOut()

    folds=list(logo.split(X_chest,y,groups))

    results={"CHEST":{"folds":[]},"LEFT":{"folds":[]},"RIGHT":{"folds":[]},"ENSEMBLE":{"folds":[]}}

    pbar=tqdm(folds,total=len(folds),desc="LOSO")

    for fold_number,(train_idx,test_idx) in enumerate(pbar,start=1):

        test_subject=groups[test_idx][0]

        chest_results=train_single_sensor_model("CHEST",X_chest[train_idx],X_chest[test_idx],y[train_idx],y[test_idx],test_subject,fold_number,CHEST_DIR)

        left_results=train_single_sensor_model("LEFT",X_left[train_idx],X_left[test_idx],y[train_idx],y[test_idx],test_subject,fold_number,LEFT_DIR)

        right_results=train_single_sensor_model("RIGHT",X_right[train_idx],X_right[test_idx],y[train_idx],y[test_idx],test_subject,fold_number,RIGHT_DIR)

        ensemble_probs=(chest_results["y_prob"]+left_results["y_prob"]+right_results["y_prob"])/3.0

        ensemble_preds=(ensemble_probs>=0.5).astype(int)

        ensemble_metrics=compute_metrics(chest_results["y_true"],ensemble_preds)

        ensemble_metrics["fold"]=fold_number
        ensemble_metrics["test_subject"]=str(test_subject)

        results["CHEST"]["folds"].append(chest_results["metrics"])
        results["LEFT"]["folds"].append(left_results["metrics"])
        results["RIGHT"]["folds"].append(right_results["metrics"])
        results["ENSEMBLE"]["folds"].append(ensemble_metrics)

        pbar.set_postfix(ensemble_f1=f"{ensemble_metrics['f1']:.4f}")

        with open(RESULTS_FILE,"w") as f:
            json.dump(results,f,indent=4)

    for model_name in ["CHEST","LEFT","RIGHT","ENSEMBLE"]:

        folds_results=results[model_name]["folds"]

        for metric in ["accuracy","precision","recall","f1","tp","fp","fn","tn"]:

            values=[x[metric] for x in folds_results]

            results[model_name][f"{metric}_mean"]=float(np.mean(values))
            results[model_name][f"{metric}_std"]=float(np.std(values))

    train_final_model("CHEST",X_chest,y,CHEST_DIR)
    train_final_model("LEFT",X_left,y,LEFT_DIR)
    train_final_model("RIGHT",X_right,y,RIGHT_DIR)

    with open(RESULTS_FILE,"w") as f:
        json.dump(results,f,indent=4)

    print(json.dumps(results,indent=4))