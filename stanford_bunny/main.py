import polyscope as ps
import trimesh

# from ..vectoradammodified import VectorAdamModified

ps.set_ground_plane_mode("shadow_only")
ps.init()

mesh = trimesh.load("assets/stanford-bunny.obj")

vertices = mesh.vertices
faces = mesh.faces
ps.register_surface_mesh("my_sphere", vertices, faces, transparency=0.2, material="ceramic")
ps.show()