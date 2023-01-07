import torch
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from model import DeepGAT,GAT
import hydra
from hydra import utils
from tqdm import tqdm
import mlflow
from utils import EarlyStopping,set_seed


def train(loader,model,optimizer,device):
    model.train()
    loss_op = torch.nn.BCEWithLogitsLoss()
    total_loss = 0
    if model.cfg['layer_loss'] == 'supervised':
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out,hs = model(data.x, data.edge_index)
            loss = loss_op(out, data.y)/(len(hs)+1)
            loss +=get_y_preds_loss(hs,data)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
    else:
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = loss_op(model(data.x, data.edge_index), data.y)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader,model,device):
    model.eval()
    ys, preds = [], []
    if model.cfg['layer_loss'] == 'supervised':
        for data in loader:
            ys.append(data.y)
            out,_ = model(data.x.to(device), data.edge_index.to(device))
            preds.append((out > 0).float().cpu())
    else:
        for data in loader:
            ys.append(data.y)
            out = model(data.x.to(device), data.edge_index.to(device))
            preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

def get_y_preds_loss(hs,data):
    loss_op = torch.nn.BCEWithLogitsLoss()
    y_pred_loss = torch.tensor(0, dtype=torch.float32,device=hs[0].device)
    for h in hs:
        h = h.mean(dim=1)
        y_pred_loss += loss_op(h,data.y)/(len(hs)+1)
    return y_pred_loss

def run(loader,model,optimizer,device,cfg):

    train_loader,test_loader = loader
    early_stopping = EarlyStopping(cfg['patience'],path=cfg['path'])

    for epoch in range(cfg['epochs']):
        loss_val = train(train_loader,model,optimizer,device)
        if early_stopping(loss_val,model) is True:
            break
    
    model.load_state_dict(torch.load(cfg['path']))
    test_acc = test(test_loader,model,device)
    return test_acc,epoch

@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    torch.cuda.empty_cache()
    print(utils.get_original_cwd())
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("output")
    mlflow.start_run()

    cfg = cfg[cfg.key]

    for key,value in cfg.items():
        mlflow.log_param(key,value)
    
    root = utils.get_original_cwd() + '/data/' + cfg['dataset']
    train_dataset = PPI(root, split='train')
    val_dataset = PPI(root, split='val')
    test_dataset = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    loader =[train_loader,test_loader]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_accs = [] 
    for i in tqdm(range(cfg['run'])):
        set_seed(i)
        if cfg['mode'] == 'original':
            model = GAT(cfg).to(device)
        else:
            model = DeepGAT(cfg).to(device)
             
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learing_late'])
        test_acc,_ = run(loader,model,optimizer,device,cfg)
        test_accs.append(test_acc)

    test_acc_ave = sum(test_accs)/len(test_accs)


    mlflow.log_metric('test_acc_min',min(test_accs))
    mlflow.log_metric('test_acc_mean',test_acc_ave)
    mlflow.log_metric('test_acc_max',max(test_accs))
    
    mlflow.end_run()
    return test_acc_ave


    
if __name__ == "__main__":
    main()