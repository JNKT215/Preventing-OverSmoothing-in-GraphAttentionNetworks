import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr
from model import DeepGAT,GAT
import hydra
from hydra import utils
from tqdm import tqdm
import mlflow
from utils import EarlyStopping,set_seed,graph_visualize

def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    if model.cfg['layer_loss'] == 'supervised':
        out_train,hs = model(data.x, data.edge_index)
        loss_train  = F.nll_loss(out_train[data.train_mask], data.y[data.train_mask])
        loss_train += get_y_preds_loss(hs,data)
    else:
        out_train = model(data.x, data.edge_index)
        loss_train  = F.nll_loss(out_train[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    if model.cfg['layer_loss'] == 'supervised':
        out_val,_ = model(data.x, data.edge_index)
    else:
        out_val = model(data.x, data.edge_index)
    loss_val = F.nll_loss(out_val[data.val_mask], data.y[data.val_mask])

    return loss_val.item()


@torch.no_grad()
def test(data,model):
    model.eval()
    if model.cfg['layer_loss'] == 'supervised':
        out,hs = model(data.x, data.edge_index)
        # check_ypred_acc(hs,data)
    else:
        out = model(data.x, data.edge_index)
    # path = utils.get_original_cwd()
    # graph_visualize(data,out,model.cfg,path)
    acc = accuracy(out,data,'test_mask')
    return acc

def accuracy(out,data,mask):
    mask = data[mask]
    acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
    return acc

def get_y_preds_loss(hs,data):
    y_pred_loss = torch.tensor(0, dtype=torch.float32,device=hs[0].device)
    for h in hs:
        h = h.mean(dim=1)
        y_pred = F.log_softmax(h, dim=-1)
        y_pred_loss += F.nll_loss(y_pred[data.train_mask], data.y[data.train_mask])

    return y_pred_loss


def run(data,model,optimizer,cfg):

    early_stopping = EarlyStopping(cfg['patience'],path=cfg['path'])

    for epoch in range(cfg['epochs']):
        loss_val = train(data,model,optimizer)
        if early_stopping(loss_val,model) is True:
            break
    
    model.load_state_dict(torch.load(cfg['path']))
    test_acc = test(data,model)
    return test_acc,epoch


@hydra.main(config_path='conf', config_name='config')
def main(cfg):

    print(utils.get_original_cwd())
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("output")
    mlflow.start_run()
    
    cfg = cfg[cfg.key]

    for key,value in cfg.items():
        mlflow.log_param(key,value)
        
    root = utils.get_original_cwd() + '/data/' + cfg['dataset']
    dataset = Flickr(root= root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    
    test_accs = [] 
    for i in tqdm(range(cfg['run'])):
        set_seed(i)
        if cfg['mode'] == 'original':
            model = GAT(cfg).to(device)
        else:
            model = DeepGAT(cfg).to(device)
            
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg["learing_late"],weight_decay=cfg['weight_decay'])
        test_acc,_ = run(data,model,optimizer,cfg)
        test_accs.append(test_acc)
            
    test_acc_ave = sum(test_accs)/len(test_accs)

        
        

    mlflow.log_metric('test_acc_min',min(test_accs))
    mlflow.log_metric('test_acc_mean',test_acc_ave)
    mlflow.log_metric('test_acc_max',max(test_accs))
    mlflow.end_run()
    return test_acc_ave

    
    

if __name__ == "__main__":
    main()