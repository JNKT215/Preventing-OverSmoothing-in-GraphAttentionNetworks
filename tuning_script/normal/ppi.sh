# Physics (original)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 exp14/train_ppi.py -m 'key=GAT_ppi' \
     'GAT_ppi.n_head=4' \
     'GAT_ppi.n_head_last=6' \
     'GAT_ppi.mode=normal' \
     'GAT_ppi.run=3' \
     'GAT_ppi.num_layer=$3' \
     'GAT_ppi.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_ppi.dropout=choice(0.,0.2,0.4,0.6,0.8)' \
     'GAT_ppi.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_ppi.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_ppi.n_hid=121' \
     'GAT_ppi.att_type=$1'\
     'GAT_ppi.layer_loss=$2'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done