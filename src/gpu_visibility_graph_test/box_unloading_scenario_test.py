import sys
import os
import numpy as np
import time
sys.path.append('home/michaelszeng/UROP-RLRG-GCS-Box_Unloading/Codebase/src/gpu_visibility_graph_test/pycuci')
import pycuci_bindings as pycuci
print(f"{dir(pycuci)}\n")


cuci_parser = pycuci.URDFParser()
cuci_parser.parse_directives("/home/michaelszeng/UROP-RLRG-GCS-Box_Unloading/Codebase/src/gpu_visibility_graph_test/box_unloading_env.yaml")

pl = cuci_parser.build_plant()
kt = pl.getKinematicTree()
links = kt.get_links()
for l in links:
    print(l.name)


t1 = time.time()
cuci_domain = pycuci.HPolyhedron()
cuci_domain.MakeBox(np.array([-3.15, -3.05433, -2.09439510239, -3.141592, -2.18166, -3.15]).T, np.array([3.15, 1.0472, 2.87979, 3.141592, 2.18166, 3.15]).T)
nodes_samples = pycuci.UniformSampleInHPolyhedraCuda([cuci_domain], cuci_domain.ChebyshevCenter(), 100000, 100)[0]
is_col_free = pycuci.CheckCollisionFreeCuda(nodes_samples, pl.getMinimalPlant())

N = 10000
nodes = nodes_samples[:, np.where(is_col_free)[0]]
nodes = nodes[:, :N]
print(f"found {nodes.shape[1]} collision-free nodes")

# TODO: make 5000000 smaller since my GPU is small; this is the batch size
# vg_cuci = pycuci.VisibilityGraph(nodes, pl.getMinimalPlant(), 0.1, 500000)  # 5000000


# Test the sampling is the same as normal sampler
from pydrake.all import (
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
)

robot_diagram_builder = RobotDiagramBuilder()
robot_model_instances = robot_diagram_builder.parser().AddModels("/home/michaelszeng/UROP-RLRG-GCS-Box_Unloading/Codebase/src/gpu_visibility_graph_test/box_unloading_env.dmd.yaml")
robot_diagram_builder_diagram = robot_diagram_builder.Build()

collision_checker_params = dict(edge_step_size=0.125)
collision_checker_params["robot_model_instances"] = robot_model_instances
collision_checker_params["model"] = robot_diagram_builder_diagram
collision_checker = SceneGraphCollisionChecker(**collision_checker_params)
ground_truth_is_col_free = collision_checker.CheckConfigsCollisionFree(nodes_samples.T)

print(is_col_free[:20])
print(ground_truth_is_col_free[:20])

print(np.equal(is_col_free, ground_truth_is_col_free)[:20])

print(nodes[:,0])
print(nodes[:,1])
print(nodes[:,2])

# compare is_col_free and ground_truth_is_col_free





import pyvista as pv
cspace_dim=6
plotter = pv.Plotter(shape=(4, 5), notebook=False)

for i in range(20):
    # Update the index each axis of the current plot represents so each of the (cspace_dim choose 3) plots is unique
    if i == 0:
        x_idx = 0
        y_idx = 1
        z_idx = 2
    else:
        if z_idx != cspace_dim-1:
            z_idx += 1
        elif y_idx != cspace_dim-2:  # and z_idx == cspace_dim-1
            y_idx += 1
            z_idx = y_idx + 1
        else:  # and z_idx == cspace_dim-1 and y_idx == cspace_dim-2
            x_idx += 1
            y_idx = x_idx + 1
            z_idx = y_idx + 1

    plotter.subplot(i // 5, i % 5)

    plotter.add_mesh(pv.PolyData(nodes[[x_idx, y_idx, z_idx], :].T), 
                    render_points_as_spheres=True, point_size=2.5)
    
    plotter.camera_position = 'xy'
    plotter.camera.azimuth = 45
    plotter.camera.elevation = 45
    plotter.show_grid()
    plotter.show_bounds(
        grid='back',
        axes_ranges=[-3.15, 3.15, -3.15, 3.15, -3.15, 3.15],
        location='outer',
        ticks='both',
        show_xlabels=False,
        show_ylabels=False,
        show_zlabels=False,
        xtitle=f"idx={x_idx}",
        ytitle=f"idx={y_idx}",
        ztitle=f"idx={z_idx}",
    )

# plotter.show()