import numpy as np
import polyscope as ps
import trimesh

from vectoradammodified import VectorAdamModified

ps.set_ground_plane_mode("shadow_only")
ps.init()

mesh = trimesh.load("assets/stanford-bunny.obj")
mesh.apply_scale(5)

# [[-0.359545  0.90386  -0.279685]] # anchor point 1 (on the ear)
# [[-0.303015  0.176095  0.221145]] # anchor point 2 (on the foot)
# [[0.300525 0.304485 0.0358  ]] # starting point 1 (on the butt)

a = [[-1, 1, -1]]
b = [[-0.5, -0.5, 0.5]]
c = [[2, 0, -0.5]]

points = [
    np.array(a),
    np.array(b),
    np.array(c)
]

points1 = [
    a,
    b,
    c
]

for i in range(3):
    ps.register_point_cloud(f"{i}", points[i], radius=0.02, color=(0,0,0))
    closest_point, distance, face_index = mesh.nearest.on_surface(points1[i])
    print(closest_point)
    ps.register_point_cloud(f"{i}_closest", np.array(closest_point), radius=0.02, color=(0,0,0))
    

vertices = mesh.vertices
faces = mesh.faces
ps.register_surface_mesh("my_sphere", vertices, faces, transparency=0.2, material="ceramic")
ps.show()