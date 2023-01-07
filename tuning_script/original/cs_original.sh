# CS (original)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 exp14/train_coauthor.py -m 'key=GAT_cs' \
     'GAT_cs.n_head=8' \
     'GAT_cs.n_head_last=1' \
     'GAT_cs.mode=original' \
     'GAT_cs.run=10' \
     'GAT_cs.num_layer=2' \
     'GAT_cs.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_cs.dropout=choice(0.,0.2,0.4,0.6,0.8)' \
     'GAT_cs.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_cs.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_cs.n_hid=choice(8, 16, 32)' \
     'GAT_cs.att_type=$1'\
     'GAT_cs.layer_loss=unsupervised'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done
