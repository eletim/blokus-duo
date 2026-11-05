import numpy as np
from blokus.ppo_generation_eval import clustered

def test_prefix_cluster_interval_keeps_paired_draws():
 # A paired +1/-1 per prefix has point .5 and difference 0; not two samples.
 a=np.tile([.5,0.],(96,1));r=clustered(a)
 assert r['point_ci95']==[.5,.5] and r['difference_ci95']==[0.,0.]

def test_cluster_support_and_reproducibility():
 a=np.array([[0.,-4],[1.,2],[.5,0]])
 r=clustered(a);assert r==clustered(a)
 assert -4<=r['difference_ci95'][0]<=r['difference_ci95'][1]<=2
