# Physics (normal)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 exp14/train_coauthor.py -m 'key=GAT_physics' \
     'GAT_physics.n_head=8' \
     'GAT_physics.n_head_last=1' \
     'GAT_physics.mode=normal' \
     'GAT_physics.run=10' \
     'GAT_physics.num_layer=$3' \
     'GAT_physics.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_physics.dropout=choice(0.,0.2,0.4,0.6,0.8)' \
     'GAT_physics.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_physics.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_physics.n_hid=5' \
     'GAT_physics.att_type=$1'\
     'GAT_physics.layer_loss=$2'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done