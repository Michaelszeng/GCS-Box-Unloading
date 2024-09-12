import sys
import os
import numpy as np
import time
sys.path.append('home/michaelszeng/UROP-RLRG-GCS-Box_Unloading/Codebase/src/gpu_visibility_graph_test/pycuci')
import pycuci_bindings as pycuci
print(f"{dir(pycuci)}\n")


cuci_parser = pycuci.URDFParser()
cuci_parser.parse_directives("/home/michaelszeng/UROP-RLRG-GCS-Box_Unloading/Codebase/src/gpu_visibility_graph_test/test_env.yaml")

pl = cuci_parser.build_plant()
kt = pl.getKinematicTree()
links = kt.get_links()
for l in links:
    print(l.name)


t1 = time.time()
cuci_domain = pycuci.HPolyhedron()
cuci_domain.MakeBox(np.array([-2,-2]).T, np.array([2,2]).T)
nodes = pycuci.UniformSampleInHPolyhedraCuda([cuci_domain], cuci_domain.ChebyshevCenter(), 10000000, 100)[0]
is_col_free = pycuci.CheckCollisionFreeCuda(nodes, pl.getMinimalPlant())

N = 10000
nodes = nodes[:, np.where(is_col_free)[0]]
nodes = nodes[:, :N]
print(f"found {nodes.shape[1]} collision-free nodes")

# TODO: make 5000000 smaller since my GPU is small; this is the batch size
vg_cuci = pycuci.VisibilityGraph(nodes, pl.getMinimalPlant(), 0.1, 500000)  # 5000000

import matplotlib.pyplot as plt

plt.scatter(nodes[0,:], nodes[1, :])
plt.show()