# Flickr (original)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 exp10/train_flickr.py -m 'key=GAT_Flickr' \
     'GAT_Flickr.n_head=8' \
     'GAT_Flickr.n_head_last=1' \
     'GAT_Flickr.mode=original' \
     'GAT_Flickr.run=10' \
     'GAT_Flickr.num_layer=5' \
     'GAT_Flickr.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_Flickr.dropout=choice(0.,0.4,0.6)' \
     'GAT_Flickr.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_Flickr.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_Flickr.n_hid=choice(8, 16, 32)' \
     'GAT_Flickr.att_type=choice(DP,SD)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done

#Flickr (normal)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 exp10/train_flickr.py -m 'key=GAT_Flickr' \
     'GAT_Flickr.n_head=8' \
     'GAT_Flickr.n_head_last=1' \
     'GAT_Flickr.mode=normal' \
     'GAT_Flickr.run=10' \
     'GAT_Flickr.num_layer=5' \
     'GAT_Flickr.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_Flickr.dropout=choice(0.,0.4,0.6)' \
     'GAT_Flickr.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_Flickr.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_Flickr.n_hid=7' \
     'GAT_Flickr.layer_loss=choice(unsupervised,supervised)' \
     'GAT_Flickr.att_type=choice(YDP,YSD)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done