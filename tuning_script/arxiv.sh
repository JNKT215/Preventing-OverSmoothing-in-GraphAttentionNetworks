# Flickr (original)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 exp10/train_arxiv.py -m 'key=GAT_Arxiv' \
     'GAT_Arxiv.n_head=8' \
     'GAT_Arxiv.n_head_last=1' \
     'GAT_Arxiv.mode=original' \
     'GAT_Arxiv.run=10' \
     'GAT_Arxiv.num_layer=5' \
     'GAT_Arxiv.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_Arxiv.dropout=choice(0.,0.4,0.6)' \
     'GAT_Arxiv.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_Arxiv.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_Arxiv.n_hid=choice(64,128,256)' \
     'GAT_Arxiv.att_type=choice(DP,SD)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done

#Flickr (normal)
IFS_BACKUP=$IFS
IFS=$'\n'
ary=("
     python3 exp10/train_arxiv.py -m 'key=GAT_Arxiv' \
     'GAT_Arxiv.n_head=8' \
     'GAT_Arxiv.n_head_last=1' \
     'GAT_Arxiv.mode=normal' \
     'GAT_Arxiv.run=10' \
     'GAT_Arxiv.num_layer=5' \
     'GAT_Arxiv.norm=choice(None,LayerNorm,BatchNorm1d)' \
     'GAT_Arxiv.dropout=choice(0.,0.4,0.6)' \
     'GAT_Arxiv.learing_late=choice(0.05,0.01,0.005,0.001)' \
     'GAT_Arxiv.weight_decay=choice(0,1E-4,5E-4,1E-3)' \
     'GAT_Arxiv.n_hid=40' \
     'GAT_Arxiv.layer_loss=choice(unsupervised,supervised)' \
     'GAT_Arxiv.att_type=choice(YDP,YSD)'\
     ")

for STR in ${ary[@]}
do
    eval "${STR}"
done