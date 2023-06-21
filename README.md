# Contrastive Learning using Random Walk (CLRW)

**Overview of the framework.** Each image is augmented twice to produce two views. The edges represent probability transitions of pair of views. Thick edges represent high probability transitions (positive pairs) that are maximized toward 1, while dotted lines correspond to low probability transitions (negative pairs), minimized toward 0.\
![alt text](https://github.com/ilyassmoummad/CLRW/blob/master/CLRW_fig.png)

to launch CLRW :\
```python3 main.py  --epochs 100 --epochs2 100 --lr 0.1 --lr2 0.1 --tau 0.4 --datadir path_for_storing_data```

To launch SimCLR, add the argument ```--simclr```

To use AutoAugment, add the argument ```--autoaugment```
