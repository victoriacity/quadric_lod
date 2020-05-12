import numpy as np 

'''
Class to represent triangle meshes.
'''
class TriangleMesh:

    def from_obj(objfile):
        with open(objfile, 'r') as f:
            verts = []
            faces = []
            edges = []
            material = None
            for line in f.readlines():
                token = line.strip().split()
                if not token:
                    continue
                if token[0] == "newmtl":
                    cur_material = {'name': token[1]}
                elif token[0] in ["Ka", "Kd"]:
                    cur_material[token[0]] = token[1:]
                elif token[0] == "usemtl":
                    # this assumes only 1 material is present in the OBJ file!
                    # ignores the name of material to be used
                    material = cur_material
                elif token[0] == "v":
                    verts.append(np.array(token[1:]).astype(np.float))
                elif token[0] == "f":
                    faces.append(np.array(token[1:]).astype(np.int))
                elif token[0] == "l":
                    edges.append(np.array(token[1:]).astype(np.int))
            verts = np.array(verts)
        return TriangleMesh(verts, faces, edges, material, name=objfile)

    def __init__(self, verts, faces, edges, material, name='', adjacency_list=True):
        self.verts = verts
        self.faces = faces
        self.edges = edges
        self.material = material
        self.is_edge_adjacency = adjacency_list
        # only after triangulation can the mesh 
        # be made into a numpy matrix
        self.faces = np.array(faces)
        # convert 1-index to 0-index
        self.faces -= 1
        self.clean_faces()
        self.name = name
        if not edges:
            if adjacency_list:
                self.__calculate_adjacency()
            else:
                self.__calculate_edges()

    def dims(self):
        return np.min(self.verts, axis=0), np.max(self.verts, axis=0)
          
    # OBJ vertices are 1-indexed
    def get_edge(self, index):
        return self.verts[self.edges[index]]

    def get_face(self, index):
        return self.verts[self.faces[index]]

    def get_faces(self):
        return self.get_face(np.arange(len(self.faces)))

    def to_obj(self, filename):
        lines = []
        with open(filename, 'w') as outfile:
            if self.material:
                lines.append("newmtl %s\n" % self.material["name"])
                if "Ka" in self.material:
                    lines.append("Ka %s\n" % ' '.join(self.material["Ka"]))
                if "Kd" in self.material:
                    lines.append("Kd %s\n" % ' '.join(self.material["Kd"]))
                lines.append("\n")
                lines.append("usemtl [%s]\n" % self.material["name"])
            for i, v in enumerate(self.verts):
                lines.append("v %s\n" % ' '.join(v.astype(np.str).tolist()))
            lines.append("\n")
            if not self.is_edge_adjacency:
                for e in self.edges:
                    lines.append("l %s\n" % ' '.join((e+1).astype(np.str).tolist()))
                lines.append("\n")
            else:
                lines.append("s off \n")
            for f in self.faces:
                if f[0] == -1:
                    continue
                lines.append("f %s\n" % ' '.join((f+1).astype(np.str).tolist()))
            lines.append("\n")
            outfile.writelines(lines)

    def __str__(self):
        return "Mesh %s: [Vertices: %d, Faces: %d, Material: %s]" \
                % (self.name, len(self.verts), len(self.faces), self.material)

    def __calculate_edges(self):
        hashset = set()
        # may work with untriangulated faces
        for f in self.faces:
            for i in range(len(f)):
                v0, v1 = f[i - 1], f[i]
                if v0 > v1:
                    v0, v1 = v1, v0
                if (v0, v1) not in hashset:
                    hashset.add((v0, v1))
                    self.edges.append([v0, v1])
        self.edges = np.array(self.edges)

    def __calculate_adjacency(self):
        hashset = set()
        self.edges = [[] for x in range(len(self.verts))]
        self.face_keys = [set() for x in range(len(self.verts))]
        for i, f in enumerate(self.faces):
            for j in range(len(f)):
                v0, v1 = f[j - 1], f[j]
                self.face_keys[v0].add(i)
                if v0 > v1:
                    v0, v1 = v1, v0
                if (v0, v1) not in hashset:
                    hashset.add((v0, v1))
                    self.edges[v1].append(v0)
                    self.edges[v0].append(v1)
        self.edges = np.array(self.edges)
        self.face_keys = np.array(self.face_keys)

    def clean_faces(self):
        faces = self.get_faces()
        vec1 = faces[:, 1, :] - faces[:, 0, :]
        vec2 = faces[:, 2, :] - faces[:, 1, :]
        norm = np.sum(np.cross(vec1, vec2) ** 2, axis=1)
        self.faces = self.faces[norm > 0, :]

    def remove_loose(self):
        p = 0
        removed = []
        for i in range(len(self.verts)):
            if len(self.face_keys[i]) == 0 or np.all(self.faces[list(self.face_keys[i])] == -1):
                removed.append(i)
                continue
            if i != p:
                for f in self.face_keys[i]:
                    self.faces[f][self.faces[f] == i] = p
            p += 1
        self.edges = np.delete(self.edges, removed)
        self.face_keys = np.delete(self.face_keys, removed)
        mask = np.array([True] * len(self.verts))
        mask[removed] = False
        self.verts = self.verts[mask]
        self.faces = self.faces[self.faces[:, 0] > -1]
        print(len(removed), "loose vertices removed.")
