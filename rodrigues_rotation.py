from scipy.spatial.transform import Rotation
import numpy as np
import polyscope as ps
import trimesh

def apply_rotation_matrix(rotation_matrix, vector):
    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

ps.init()

sphere = trimesh.primitives.Sphere(radius=1)

vertices = sphere.vertices
faces = sphere.faces

ps.register_surface_mesh("my_sphere", vertices, faces, transparency=0.2, material="ceramic")

# rotate from b to a
a = np.array([1, 0, 0])
b = np.array([0, 0, 1])

v1 = np.array([1, 0, 0])

b_cloud = ps.register_point_cloud("Initial b", np.reshape(b, (1, 3)), radius=0.02, color=(0.48, 0.99, 0))
b_cloud.add_vector_quantity("Initial Vector", np.reshape(v1, (1, 3)), vectortype='ambient', enabled=True, color=(0.48, 0.99, 0))

ps.show()

rotation, rssd = Rotation.align_vectors(np.reshape(a, (1, 3)), np.reshape(b, (1, 3)))

rotation_matrix = rotation.as_matrix()
v2 = apply_rotation_matrix(rotation_matrix, v1)

rotated_cloud = ps.register_point_cloud("Rotated b", np.reshape(a, (1, 3)), radius=0.02, color=(0.48, 0.99, 0))

v2_reshaped = np.reshape(v2, (1, 3))
rotated_cloud.add_vector_quantity("Rotated Vector", v2_reshaped, vectortype='ambient', enabled=True, color=(0.48, 0.99, 0))

ps.show()

# rotate from b to a
a = np.array([0, 1, 0])
b = np.array([1, 0, 0])

v1 = v2_reshaped.reshape(3)

rotation, rssd = Rotation.align_vectors(np.reshape(a, (1, 3)), np.reshape(b, (1, 3)))

rotation_matrix = rotation.as_matrix()
v2 = apply_rotation_matrix(rotation_matrix, v1)

rotated_cloud_2 = ps.register_point_cloud("Rotated b 2", np.reshape(a, (1, 3)), radius=0.02, color=(0.48, 0.99, 0))

v2_reshaped = np.reshape(v2, (1, 3))
rotated_cloud_2.add_vector_quantity("Rotated Vector", v2_reshaped, vectortype='ambient', enabled=True, color=(0.48, 0.99, 0))

ps.show()

# rotate from b to a
a = np.array([0, 0, 1])
b = np.array([0, 1, 0])

v1 = v2_reshaped.reshape(3)

rotation, rssd = Rotation.align_vectors(np.reshape(a, (1, 3)), np.reshape(b, (1, 3)))

rotation_matrix = rotation.as_matrix()
v2 = apply_rotation_matrix(rotation_matrix, v1)

rotated_cloud_2 = ps.register_point_cloud("Rotated b 3", np.reshape(a, (1, 3)), radius=0.02, color=(0.48, 0.99, 0))

v2_reshaped = np.reshape(v2, (1, 3))
rotated_cloud_2.add_vector_quantity("Rotated Vector", v2_reshaped, vectortype='ambient', enabled=True, color=(0.48, 0.99, 0))

ps.show()