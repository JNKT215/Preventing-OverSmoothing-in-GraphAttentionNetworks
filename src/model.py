import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import DeepGATConv,GATConv

class DeepGAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()
        
        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        if self.cfg['task'] == 'Transductive':
            self.inconv = DeepGATConv(in_channels=cfg['n_feat'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_convs.append(DeepGATConv(in_channels=cfg['n_hid']*cfg['n_head'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num']))
            self.outconv = DeepGATConv(in_channels=cfg['n_hid']*cfg['n_head'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg['att_type'],class_num=cfg['class_num'])
        elif self.cfg['task'] == 'Inductive':
            self.inconv = DeepGATConv(cfg['n_feat'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num'])
            self.in_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_head'] * cfg['n_hid'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_convs.append(DeepGATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg['att_type'],class_num=cfg['class_num']))
                self.mid_lins.append(torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_head'] * cfg['n_hid']))
            self.outconv = DeepGATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg['att_type'],class_num=cfg['class_num'])
            self.out_lin = torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_class'])

    def forward(self, x, edge_index):
        if self.cfg['layer_loss'] == 'supervised':
            hs = []
            if self.cfg['task'] == 'Transductive':
                x = F.dropout(x, p=self.dropout, training=self.training)
                x= self.inconv(x,edge_index)
                x = self.in_norm(x)
                hs.append(x.view(-1,self.inconv.heads,self.inconv.out_channels))
                x = F.elu(x)
                for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = mid_conv(x, edge_index)
                    x = mid_norm(x)
                    hs.append(x.view(-1,mid_conv.heads,mid_conv.out_channels))
                    x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.outconv(x,edge_index)
                x = self.out_norm(x)
                x = F.log_softmax(x, dim=-1)
            elif self.cfg['task'] == 'Inductive':
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                hs.append(x.view(-1,self.inconv.heads,self.inconv.out_channels))
                x = F.elu(x)
                for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                    x = mid_conv(x, edge_index) + mid_lin(x)
                    x = mid_norm(x)
                    hs.append(x.view(-1,mid_conv.heads,mid_conv.out_channels))
                    x = F.elu(x)          
                x = self.outconv(x, edge_index) + self.out_lin(x)
                x = self.out_norm(x)
            return x,hs
        else:
            if self.cfg['task'] == 'Transductive':
                x = F.dropout(x, p=self.dropout, training=self.training)
                x= self.inconv(x,edge_index)
                x = self.in_norm(x)
                x = F.elu(x)
                for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = mid_conv(x, edge_index)
                    x = mid_norm(x)
                    x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.outconv(x,edge_index)
                x = self.out_norm(x)
                x = F.log_softmax(x, dim=-1)
            elif self.cfg['task'] == 'Inductive':
                x = self.inconv(x, edge_index) + self.in_lin(x)
                x = self.in_norm(x)
                x = F.elu(x)
                for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                    x = mid_conv(x, edge_index) + mid_lin(x)
                    x = mid_norm(x)
                    x = F.elu(x)
                x = self.outconv(x, edge_index) + self.out_lin(x)
                x = self.out_norm(x)
            return x

class GAT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.dropout = cfg['dropout']
        self.cfg = cfg
        self.mid_norms = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.mid_lins = nn.ModuleList()

        if cfg['norm'] == 'LayerNorm':
            self.in_norm = nn.LayerNorm(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.LayerNorm(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.LayerNorm(cfg['n_class'])
        elif cfg['norm'] == 'BatchNorm1d':
            self.in_norm = nn.BatchNorm1d(cfg['n_hid']*cfg['n_head'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.BatchNorm1d(cfg['n_hid']*cfg['n_head']))
            self.out_norm = nn.BatchNorm1d(cfg['n_class'])
        else:
            self.in_norm = nn.Identity()
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_norms.append(nn.Identity())
            self.out_norm = nn.Identity()
        
        if self.cfg['task'] == 'Transductive':
            self.inconv = GATConv(in_channels=cfg['n_feat'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_convs.append(GATConv(in_channels=cfg['n_hid']*cfg['n_head'],out_channels=cfg['n_hid'], heads=cfg['n_head'], dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"]))
            self.outconv = GATConv(in_channels=cfg['n_hid']*cfg['n_head'], out_channels=cfg['n_class'], heads=cfg['n_head_last'], concat=False,dropout=cfg['n_layer_dropout'],attention_type=cfg["att_type"])
        elif self.cfg['task'] == 'Inductive':
            self.inconv = GATConv(cfg['n_feat'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg["att_type"])
            self.in_lin = torch.nn.Linear(cfg['n_feat'], cfg['n_head'] * cfg['n_hid'])
            for _ in range(1,cfg["num_layer"]-1):
                self.mid_convs.append(GATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_hid'], heads=cfg['n_head'],attention_type=cfg["att_type"]))
                self.mid_lins.append(torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_head'] * cfg['n_hid']))
            self.outconv = GATConv(cfg['n_head'] * cfg['n_hid'], cfg['n_class'], heads=cfg['n_head_last'],concat=False,attention_type=cfg["att_type"])
            self.out_lin = torch.nn.Linear(cfg['n_head'] * cfg['n_hid'], cfg['n_class'])
    
    def forward(self, x, edge_index):
        if self.cfg['task'] == 'Transductive':
            x = F.dropout(x, p=self.dropout, training=self.training)
            x= self.inconv(x,edge_index)
            x = self.in_norm(x)
            x = F.elu(x)
            for mid_conv,mid_norm in zip(self.mid_convs,self.mid_norms):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = mid_conv(x, edge_index)
                x = mid_norm(x)
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.outconv(x,edge_index)
            x = self.out_norm(x)
            x = F.log_softmax(x, dim=-1)
        elif self.cfg['task'] == 'Inductive':
            x = F.elu(self.inconv(x, edge_index) + self.in_lin(x))
            x = self.in_norm(x)
            for mid_conv,mid_lin,mid_norm in zip(self.mid_convs,self.mid_lins,self.mid_norms):
                x = F.elu(mid_conv(x, edge_index) + mid_lin(x))
                x = mid_norm(x)
            x = self.outconv(x, edge_index) + self.out_lin(x)
            x = self.out_norm(x)
        return x