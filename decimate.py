import os
import numpy as np
from tqdm import tqdm

'''
Function aliases
'''
dist = lambda v1, v2: np.sqrt(np.sum((v1 - v2) ** 2))
vlen = np.vectorize(len)
vsetitem = np.vectorize(lambda x: list(x)[0])
normal = lambda f: np.cross(f[:, 1, :] - f[:, 0, :], f[:, 2, :] - f[:, 0, :])

def standardize_plane(faces):
    '''
    Converts planes defined by a 3 points (3x3 matrix with each ROW as a point)
    into standardized format Ax+By+Cz+D=0 with A^2+B^2+C^2=1.
    Also calculates the area of each triangle face.
    '''
    norm = normal(faces)
    area = np.sqrt(np.sum(norm ** 2, axis=1)) / 2
    # normalize
    norm /= np.sqrt(np.sum(norm ** 2, axis=1).reshape(-1, 1))
    # compute intercept, D=-(Ax0+bY0+CZ0)
    intercept = -np.sum(norm * faces[:, 0, :], axis=1).reshape(-1, 1)
    return np.hstack([norm, intercept]), area


def merge_vert(mesh, i, j):
    '''
    Merges vertex j into vertex i of MESH.
    To avoid recalculating all vertex indices, vertex J
    is not removed from the mesh, instead all of its edges and
    faces are moved into vertex I.
    '''
    edges = set(mesh.edges[i]) | set(mesh.edges[j])
    edges.discard(i)
    edges.discard(j)
    faces = mesh.face_keys[j] 
    # Reassign edge neighbors of i
    for e in mesh.edges[j]:
        s = set(mesh.edges[e])
        s.discard(j)
        s.add(i)
        mesh.edges[e] = list(s)
    mesh.edges[i] = list(edges)
    mesh.edges[j] = []
    # the faces of the merged vertice
    # is the XOR between the faces
    # of two vertices (union - intersection)
    mesh.faces[list(mesh.face_keys[i] & mesh.face_keys[j])] = -1
    for f in mesh.face_keys[j] - mesh.face_keys[i]:
        mesh.faces[f][mesh.faces[f] == j] = i
    mesh.face_keys[i] = mesh.face_keys[i] ^ mesh.face_keys[j]
    mesh.face_keys[j] = set()

def quadric_error(verts, quadric, weights, add_bias=True):
    '''
    Computes the error for each vertex based on error quadrics
    and vertex coordinates.
    '''
    if add_bias:
        verts = np.hstack([verts, np.ones((verts.shape[0], 1))])
    verts = verts.reshape(-1, 4, 1)
    ret = weights.reshape(-1, 1, 1) * (np.transpose(verts, (0, 2, 1)) @ quadric @ verts)
    return ret

def solve_contraction(verts, pairs, quadrics, weights, eps=1e-3):
    '''
    Solves for the optimal contraction positions for vertex pairs
    using quadric error metrics.
    '''
    dims = (np.min(verts, axis=0), np.max(verts, axis=0))
    # solve for vbar using eq. 1
    qbar = np.sum(quadrics[pairs], axis=1)
    weights = weights.reshape(-1, 1)
    weights_tot = np.sum(weights[pairs], axis=1)
    coeffs = qbar.copy()
    coeffs[:, 3, :] = 0
    coeffs[:, 3, 3] = 1
    # compute optimal positions
    det = np.abs(np.linalg.det(coeffs))
    vbar = verts[pairs[:, 0]] * weights[pairs[:, 0]] + verts[pairs[:, 1]] * weights[pairs[:, 1]]
    vbar /= weights_tot
    vbar = np.hstack([vbar, np.ones((vbar.shape[0], 1))])
    vbar[det > eps] = np.linalg.inv(coeffs[det > eps])[:, :, 3]
    mask = np.any(vbar[:, :3] < dims[0], axis=1) | np.any(vbar[:, :3] > dims[1], axis=1)
    vbar[mask, :3] = (verts[pairs[mask, 0]] * weights[pairs[mask, 0]] \
        + verts[pairs[mask, 1]] * weights[pairs[mask, 1]]) / weights_tot[mask]
    vbar[mask, 3] = 1
    # compute decimation loss
    loss = quadric_error(vbar, qbar, weights_tot, add_bias=False)
    return vbar, qbar, loss.ravel()

class QuadricDecimator: 
    
    def __init__(self, mesh, threshold, use_area=True, boundary_weight=10, scan=0):
        self.mesh = mesh
        self.pairs = self._find_valid_pairs(threshold)
        self.scan = scan
        planes, area = standardize_plane(mesh.get_faces())
        if use_area:
            # calculate face weights by normalizing using
            # the average surfac area
            self.faceweights = area / np.mean(area)
            # calculate vertex weights
            self.vertexweights = np.array([np.mean(self.faceweights[list(x)]) \
                 for x in mesh.face_keys])
            # set weights for loose vertices to be 0
            self.vertexweights[np.isnan(self.vertexweights)] = 0
        else:
            self.faceweights = np.ones(area.shape)
            self.vertexweights = np.ones(len(mesh.verts))
        # calculate quadrics
        self.quadrics = self._calc_quadric(planes)
        if boundary_weight > 0:
            self._add_boundary_constraints(boundary_weight)
        self.vbar, self.qbar, loss = solve_contraction(mesh.verts,
             self.pairs, self.quadrics, self.vertexweights)
        # if loss of two vertex pairs are numerically the same
        # break ties using the index
        self.heap = np.array(list(zip(loss, np.arange(len(loss)))),
            dtype=[('loss', float), ('index', int)])
        self.heap = np.sort(self.heap, order='loss')
        self.loss_history = []
        self.savename = os.path.splitext(os.path.basename(args.input))[0]
        self.savename += "-t%s-a%d-b%s" % (threshold, use_area, boundary_weight)

    def run(self, niter):
        if self.scan:
            os.makedirs(self.savename)
        with tqdm(total=(len(self.heap) if self.scan else niter), unit='pairs') as t:
            i = 0
            nvert = len(self.heap)
            while len(self.heap) > 0:
                if not self.scan and i >= niter:
                    break
                if self.scan and i > niter and i % self.scan == 0:
                    outfile = os.path.join(self.savename, "%d.obj" % i)
                    self.save_mesh(outfile, remove_loose=False)
                self.loss_history.append(self.heap['loss'][0])
                self._decimate()
                dv = nvert - len(self.heap)
                t.update(dv if self.scan else 1)
                nvert -= dv
                i += 1
                
    def save_mesh(self, savename, remove_loose=True):
        if remove_loose:
            mesh.remove_loose()
        mesh.to_obj(savename)

    def _find_valid_pairs(self, threshold):
        # threshold is relative to the average extent of the mesh
        pairs = []
        avgdim = np.mean(self.mesh.dims()[1] - self.mesh.dims()[0])
        for i in range(len(self.mesh.verts)):
            # vectorize the inner loop
            sqdists = np.sum((self.mesh.verts - self.mesh.verts[i]) ** 2, axis=1)
            sqdists[self.mesh.edges[i]] = 0
            neighbors, = np.where(sqdists < (avgdim * threshold) ** 2)
            neighbors = neighbors[neighbors > i]
            neighbors = np.vstack([neighbors, np.ones(len(neighbors)) * i])
            pairs.append(neighbors.T)
        return np.vstack(pairs).astype(np.int)

    def _calc_quadric(self, planes):
        # first calculate all gram matrices
        # for plane equations
        gram = planes.reshape(-1, 4, 1) @ planes.reshape(-1, 1, 4)
        # aggregate error quadric for each vertex
        quadric = np.zeros((len(self.mesh.verts), 4, 4))
        for i, f in enumerate(self.mesh.faces):
            quadric[f] += gram[i] * self.faceweights[i]
        return quadric

    def _add_boundary_constraints(self, boundary_weight):
        # Make the face weight of a boundary constraint as
        # BOUNDARY_WEIGHTS * the maximum of all face weights
        weight = np.max(self.faceweights) * boundary_weight
        ne = vlen(self.mesh.edges)
        nf = vlen(self.mesh.face_keys)
        verts_bound = np.where(ne > nf)[0]
        if len(verts_bound) == 0:
            return
        print(len(verts_bound), "boundary vertices found.")
        verts_bound_set = set(verts_bound.tolist())
        # find adjacent bound vertices for each vertex to determine boundary faces
        # use a prefix sum to vectorize
        neighbors_bound = [set(x) & verts_bound_set for x in self.mesh.edges[verts_bound]]
        prefixes = vlen(neighbors_bound)
        v0 = np.repeat(verts_bound, prefixes)
        prefixes = np.cumsum(prefixes)
        v1 = np.hstack([list(x) for x in neighbors_bound]).astype(np.int)
        neighbor_faces = vsetitem(self.mesh.face_keys[v0] & self.mesh.face_keys[v1])
        # calculate normal faces for each neighboring face
        edge = self.mesh.verts[v0] - self.mesh.verts[v1]
        nface = normal(self.mesh.get_face(neighbor_faces))
        nbound = np.cross(edge, nface)
        nbound /= np.sqrt(np.sum(nbound ** 2, axis=1).reshape(-1, 1))
        intercept = -np.sum(nbound * self.mesh.verts[v0], axis=1).reshape(-1, 1)
        bound_planes = np.hstack([nbound, intercept])
        gram = bound_planes.reshape(-1, 4, 1) @ bound_planes.reshape(-1, 1, 4)
        for i, g in zip(v0, gram):
            self.quadrics[i] += g * weight
            #self.vertexweights[i] += weight # also update vertex weights

    def _decimate(self):
        _, i = self.heap[0]
        self.heap = self.heap[1:]
        v0, v1 = self.pairs[i]
        self.mesh.verts[v0] = self.vbar[i, :3]
        merge_vert(self.mesh, v0, v1)
        # remove all v1 indices from pairs
        self.pairs[self.pairs[:, 0] == v1, 0] = v0
        self.pairs[self.pairs[:, 1] == v1, 1] = v0
        # find elements in heap to update
        heap_indices = (self.pairs[self.heap['index'], 0] == v0) | (self.pairs[self.heap['index'], 1] == v0)
        pair_indices = self.heap['index'][heap_indices]
        #if len(pair_indices) == 0:
        self.heap = self.heap[~heap_indices]
        # remove duplicate pairs
        self.pairs[pair_indices] = np.sort(self.pairs[pair_indices], axis=1)   
        # removing the vertex may not result in any pair to be processed 
        if len(pair_indices) == 0:
            return 
        _, args = np.unique(self.pairs[pair_indices], axis=0, return_index=True)
        pair_indices = pair_indices[args]
        # update quadrics
        self.quadrics[v0] = self.qbar[i]
        self.quadrics[v1] = self.qbar[i]
        # recalculate vbar and qbar
        vbar_new, qbar_new, loss_new \
            = solve_contraction(self.mesh.verts, self.pairs[pair_indices], self.quadrics, self.vertexweights)
        self.vbar[pair_indices] = vbar_new
        self.qbar[pair_indices] = qbar_new
        newheap = np.array(list(zip(loss_new, pair_indices)), dtype=[('loss', float), ('index', int)])
        self.heap = np.sort(np.concatenate([self.heap, newheap]), order='loss')

if __name__ == "__main__":
    import argparse
    from mesh import TriangleMesh
    parser = argparse.ArgumentParser(
        description='Generates LOD mesh based on quadric error metric with heuristics')
    parser.add_argument('input', type=str, help='Path to input mesh file')
    parser.add_argument('n', type=int, 
        help='Number of decimation iteration')
    parser.add_argument('-a', '--area', action="store_false", 
        help='Turn off surface area heuristic')
    parser.add_argument('-b', '--boundary', type=float, default=5, 
        help="Boundary constraint factor, use 0 to turn off")
    parser.add_argument('-t', '--threshold', type=float, default=0.05, 
        help="Pair decimation threshold relative to mean extent of the mesh")
    parser.add_argument('-o', '--output', type=str, default=None, 
        help="Path to output file")
    parser.add_argument('-p', '--plot', action="store_true",
        help="Plots loss history")
    parser.add_argument('-s', '--scan', type=int, default=0, 
        help="Scan the entire loss curve, save model every [-s] steps after [n] steps")
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.splitext(os.path.basename(args.input))[0] + "_lod.obj"
    mesh = TriangleMesh.from_obj(args.input)
    print(mesh)
    decimator = QuadricDecimator(mesh, args.threshold, args.area, args.boundary, args.scan)
    decimator.run(args.n)
    if not args.scan:
        decimator.save_mesh(args.output)
    if args.scan or args.plot:
        import matplotlib.pyplot as plt
        plt.plot(np.log10(np.array(decimator.loss_history) + 1))
        plt.xlabel("iterations")
        plt.ylabel("log1p loss")
        plt.show()


