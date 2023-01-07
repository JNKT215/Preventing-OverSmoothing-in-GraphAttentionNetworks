import torch
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import os
class EarlyStopping():
    def __init__(self,patience,path="checkpoint.pt"):
        self.best_loss_score = None
        self.loss_counter =0
        self.patience = patience
        self.path = path
        self.val_loss_min =None
        
    def __call__(self,loss_val,model):
        if self.best_loss_score is None:
            self.best_loss_score = loss_val
            self.save_best_model(model,loss_val)
        elif self.best_loss_score > loss_val:
            self.best_loss_score = loss_val
            self.loss_counter = 0
            self.save_best_model(model,loss_val)
        else:
            self.loss_counter+=1
            
        if self.loss_counter == self.patience:
            return True
        
        return False
    def save_best_model(self,model,loss_val):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = loss_val

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def concat_path(original_path,add_path):
    return os.path.normpath(os.path.join(original_path, add_path))

def graph_visualize(data,h,cfg,path):
    save_dir =  f"{path}/exp14/graph/{cfg['dataset']}/{cfg['mode']}"
    # save_dir =  f"{path}/exp14/graph/{cfg['dataset']}/{cfg['mode']}/{cfg['split']}"
    file_name = f"/{cfg['dataset']}_{cfg['att_type']}_{cfg['num_layer']}_{cfg['layer_loss']}.png"

    y_label = data.y.clone()
    y_label = y_label.to('cpu').detach().numpy().copy()
    colors = ['red', 'blue', 'green', 'pink', 'purple', 'black', 'orange', 'yellow', 'tomato', 'lime', 'olive','aqua','chocolate','blueviolet','goldenrod','aquamarine4']

    #hyper parameter
    learning = [200]
    per = [30]
    x = h.to('cpu').detach().numpy().copy()
    for i in learning:
        for j in per:
            tsne = TSNE(n_components=2, perplexity=j, learning_rate=i)
            x = h.to('cpu').detach().numpy().copy()
            X_tsne = tsne.fit_transform(x)
            fig, ax = plt.subplots()
            ax.set_xlim(X_tsne[:, 0].min()-1, X_tsne[:, 0].max() + 1)
            ax.set_ylim(X_tsne[:, 0].min()-1, X_tsne[:, 0].max() + 1)   

            #plot
            for i in range(len(x)):
                ax.scatter(X_tsne[i,0], X_tsne[i, 1], c=colors[y_label[i]])

    plt.title(cfg['dataset'])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir+file_name)
    # plt.show()
    plt.clf()
    plt.close()


def check_train_label_per(data):
    cnt = 0
    for i in data.train_mask:
        if i == True:
            cnt+=1

    train_mask_label = cnt
    labels_num = len(data.train_mask)
    train_label_percent = train_mask_label/labels_num

    print(f"train_mask_label:{cnt},labels_num:{labels_num},train_label_percent:{train_label_percent}")



