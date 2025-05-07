import pulp
import numpy as np


class Node:
    def __init__(self, data):
        self.data = data  # Mesh or set[Mesh], this has no substantial impact on ILP construction
        self.detected = False
        self.neighbors = set()  # Storing child nodes: Node
        self.force = 0  # Integer, the force applied by the node itself
        self.forces = None if isinstance(data, Mesh) else {}
        self.mass = data.mass if isinstance(data, Mesh) else 0  # Already taken the square root and rounded down
        self.static = data.static if isinstance(data, Mesh) else False  # If it is a "static" node, you can set the mass to a very large value

    @classmethod
    def pass_forward(cls, heads):
        """
        heads: List of nodes with in-degree 0
        Returns: A set, representing all the nodes that can be pushed in one ILP solution
        """

        # 1) Collect all nodes: Start from 'heads' and perform DFS/BFS traversal
        visited = set()
        def dfs(u):
            if u in visited:
                return
            visited.add(u)
            for w in u.neighbors:
                dfs(w)
        for h in heads:
            dfs(h)
        all_nodes = list(visited)
        for node in all_nodes:
            if node.force < 0:
                node.mass -= node.force
                node.force = 0

        # 2) Collect edges (v -> w)
        edges = []
        for v in all_nodes:
            for w in v.neighbors:
                edges.append((v, w))

        # 3) Build ILP problem: Maximize sum x[v]
        prob = pulp.LpProblem("PassForwardILP", pulp.LpMaximize)

        # Set a large constant M, used for slack in "residual force"
        # Depends on the actual problem scale
        M = 10000

        # 4) Define variables
        #    x[v]: 0/1, indicates if node v is pushed
        #    netForce[v]: >=0, net force on node v
        #    leftover[v]: >=0, leftover distributable force of node v
        #    f_{v->w}: >=0, force allocated by v to w
        x = {}
        netForce = {}
        leftover = {}
        f = {}  # f[(v,w)]

        for v in all_nodes:
            if v.static:
                # If v is a static node, force x[v] = 0
                x[v] = pulp.LpVariable(f"x_static_{id(v)}", cat=pulp.LpBinary)
            else:
                x[v] = pulp.LpVariable(f"x_{id(v)}", cat=pulp.LpBinary)

            netForce[v] = pulp.LpVariable(f"NF_{id(v)}", lowBound=0)
            leftover[v] = pulp.LpVariable(f"LF_{id(v)}", lowBound=0)

        for (v, w) in edges:
            f[(v,w)] = pulp.LpVariable(f"f_{id(v)}_{id(w)}", lowBound=0)

        # 5) Objective function: Maximize sum x[v]
        prob += pulp.lpSum(x[v] for v in all_nodes)

        # 6) Add constraints

        # 6.1 Static nodes force x[v] = 0
        for v in all_nodes:
            if v.static:
                prob.addConstraint( x[v] == 0, f"static_{id(v)}" )

        # 6.2 Downward closure: If v is selected => w is also selected (v->w)
        #     x[w] >= x[v]
        for (v, w) in edges:
            prob.addConstraint( x[w] >= x[v], f"desc_closure_{id(v)}_{id(w)}" )

        # 6.3 netForce[v] = v.force + sum_{u->v} f[u->v]
        #    First build in_neighbors[v]
        in_neighbors = { nd: [] for nd in all_nodes }
        for (u, v) in edges:
            in_neighbors[v].append(u)

        for v in all_nodes:
            prob.addConstraint(
                netForce[v] == v.force + pulp.lpSum(f[(u,v)] for u in in_neighbors[v]),
                f"netForceDef_{id(v)}"
            )

        # 6.4 If x[v]=1 => netForce[v] >= mass[v]
        for v in all_nodes:
            prob.addConstraint(
                netForce[v] >= v.mass * x[v],
                f"minForce_{id(v)}"
            )

        # 6.5 leftover[v] = max(netForce[v]-mass[v], 0)
        #    Use inequalities + big M trick:
        #    leftover[v] >= netForce[v] - mass[v]
        #    leftover[v] <= netForce[v] - mass[v] + M*(1 - x[v])
        #    leftover[v] >= 0   (already in variable definition)
        for v in all_nodes:
            prob.addConstraint(
                leftover[v] >= netForce[v] - v.mass,
                f"leftover_lb_{id(v)}"
            )
            prob.addConstraint(
                leftover[v] <= netForce[v] - v.mass + M*(1 - x[v]),
                f"leftover_ub_{id(v)}"
            )

        # 6.6 Force allocated to child nodes <= leftover[v]
        #    sum_{w in neighbors[v]} f[v->w] <= leftover[v]
        out_map = {}
        for v in all_nodes:
            out_map[v] = []
        for (v, w) in edges:
            out_map[v].append(w)

        for v in all_nodes:
            prob.addConstraint(
                pulp.lpSum(f[(v,w)] for w in out_map[v]) <= leftover[v],
                f"sum_outflow_{id(v)}"
            )
            prob.addConstraint(
                pulp.lpSum(f[(v,w)] for w in out_map[v]) <= x[v] * M,
                f"can_outflow_{id(v)}"
            )

        # 6.7 Force on edge must be 0 if the child node is not selected
        #     f[v->w] <= M * x[w]
        for (v, w) in edges:
            prob.addConstraint(
                f[(v,w)] <= M * x[w],
                f"edge_flow_{id(v)}_{id(w)}"
            )

        # 7) Solve
        solver = pulp.PULP_CBC_CMD(msg=0)
        result = prob.solve(solver)

        # for var in prob.variables():
        #     print(f'{var.getName()} = {pulp.value(var)}')

        # 8) If feasible (or optimal), collect nodes with x[v]=1
        pushable_set = set()
        if pulp.value(prob.status) == 1:  # LpStatusOptimal
            for v in all_nodes:
                xv_val = pulp.value(x[v])
                if xv_val is not None and xv_val >= 0.5:
                    pushable_set.add(v)

        return pushable_set

    def __repr__(self):
        return f"Node({self.data}, mass={self.mass}, force={self.force}, static={self.static})"


class Mesh:
    def __init__(self, pos, shape: np.ndarray, static, name=''):
        self.pos = pos
        self.shape = shape
        self.name = name
        self.edges = {
            (-1, 0): [],
            (1, 0): [],
            (0, 1): [],
            (0, -1): []
        }
        self.node = None
        self.static = static
        self.mass = 0
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j] == '.':
                    continue
                for di, dj in self.edges:
                    if not (0 <= i + di < shape.shape[0] and 0 <= j + dj < shape.shape[1]):
                        self.edges[(di, dj)].append((i + di, j + dj))
                    elif shape[i + di][j + dj] == '.':
                        self.edges[(di, dj)].append((i + di, j + dj))
                self.mass += 1
        self.mass = int(self.mass ** 0.5)

    def __gen_mask(self, shape, bias):
        mask = np.full(shape, '.', dtype=str)
        pos = self.pos[0] + bias[0], self.pos[1] + bias[1]
        i1 = max(pos[0], 0)
        i2 = min(pos[0] + self.shape.shape[0], mask.shape[0])
        j1 = max(pos[1], 0)
        j2 = min(pos[1] + self.shape.shape[1], mask.shape[1])
        mask[i1:i2, j1:j2] = self.shape[i1 - pos[0]:i2 - pos[0],
                             j1 - pos[1]:j2 - pos[1]]
        return mask

    def place(self, grid=None, mesh_map=None, bias=(0, 0)):
        mask = self.__gen_mask((mesh_map if grid is None else grid).shape, bias)
        if grid is not None:
            grid[mask != '.'] = mask[mask != '.']
        if mesh_map is not None:
            mesh_map[mask != '.'] = self

    def remove(self, grid=None, mesh_map=None, bias=(0, 0)):
        mask = self.__gen_mask((mesh_map if grid is None else grid).shape, bias)
        if grid is not None:
            grid[mask != '.'] = '.'
        if mesh_map is not None:
            mesh_map[mask != '.'] = None

    def detect(self, mesh_map, di, dj, bias=(0, 0)):
        self.node.detected = True
        for ci, cj in self.edges[(di, dj)]:
            i = self.pos[0] + ci + bias[0]
            j = self.pos[1] + cj + bias[1]
            if not (0 <= i < mesh_map.shape[0] and 0 <= j < mesh_map.shape[1]):
                continue
            if mesh_map[i][j] is not None:
                self.node.neighbors.add(mesh_map[i][j].node)

    @classmethod
    def extend_map(cls, mesh_map):
        i1, j1 = 0, 0
        i2, j2 = mesh_map.shape
        meshes = set()
        for i in range(mesh_map.shape[0]):
            for j in range(mesh_map.shape[1]):
                if mesh_map[i, j] is None or mesh_map[i, j] in meshes:
                    continue
                meshes.add(mesh_map[i, j])
                i1 = min(mesh_map[i, j].pos[0], i1)
                i2 = max(mesh_map[i, j].pos[0] + mesh_map[i, j].shape.shape[0], i2)
                j1 = min(mesh_map[i, j].pos[1], j1)
                j2 = max(mesh_map[i, j].pos[1] + mesh_map[i, j].shape.shape[0], j2)
        bias = -i1, -j1
        new_map = np.full((i2 - i1, j2 - j1), None, dtype=object)
        for mesh in meshes:
            mesh.place(mesh_map=new_map, bias=bias)
        # print(new_map)
        return new_map, bias

    @classmethod
    def detect_all(cls, mesh_map, di, dj):
        mesh_map, bias = cls.extend_map(mesh_map)
        for i in range(mesh_map.shape[0]):
            for j in range(mesh_map.shape[1]):
                if mesh_map[i][j] is not None:
                    mesh_map[i][j].node = Node(mesh_map[i][j])
        for i in range(mesh_map.shape[0]):
            for j in range(mesh_map.shape[1]):
                if mesh_map[i][j] is not None:
                    if mesh_map[i][j].node.detected:
                        continue
                    mesh_map[i][j].detect(mesh_map, di, dj, bias)

    def move(self, grid, mesh_map, di, dj):
        block_item = set()
        for i, j in self.edges[(di, dj)]:
            if (0 <= self.pos[0] + i < mesh_map.shape[0] and 0 <= self.pos[1] + j < mesh_map.shape[1] and
                mesh_map[self.pos[0] + i][self.pos[1] + j] is not None):
                block_item.add(mesh_map[self.pos[0] + i][self.pos[1] + j])
        if len(block_item) > 0:
            return block_item
        self.remove(grid, mesh_map)
        self.pos = (self.pos[0] + di, self.pos[1] + dj)
        self.place(grid, mesh_map)
        return block_item

    @classmethod
    def build_dag(cls, mesh_map):
        """
        Treat all mesh.node on mesh_map as nodes in a graph, build a directed graph based on node.neighbors;
        If a cycle appears in the graph, the corresponding strongly connected component is merged into a new supernode, forming a DAG.

        Returns:
            Returns a list containing all "supernodes" with an in-degree of 0 (each supernode is a Node object),
            In these supernodes, node.data might be a set[Mesh], representing multiple Meshes merged together.
        """

        # ---------------------------------------------------------------------
        # 1. Collect all nodes
        # ---------------------------------------------------------------------
        all_nodes = []
        for i in range(mesh_map.shape[0]):
            for j in range(mesh_map.shape[1]):
                if mesh_map[i][j] is not None:
                    all_nodes.append(mesh_map[i][j].node)

        # If there are no nodes, return directly
        if not all_nodes:
            return []

        # ---------------------------------------------------------------------
        # 2. Use Tarjan's algorithm (or other algorithms) to find strongly connected components
        # ---------------------------------------------------------------------
        index_counter = [0]  # Record DFS visit order
        stack = []
        on_stack = set()
        scc_id = {}  # node -> scc index
        indexes = {}  # node -> DFS visit order
        lowlink = {}  # node -> The smallest index this node can trace back to
        current_scc_count = [0]  # Number of strongly connected components found

        def strongconnect(node):
            # Set the index and lowlink of this node
            indexes[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            # Iterate over all possible neighbors
            for nxt in node.neighbors:
                if nxt not in indexes:  # Not visited
                    strongconnect(nxt)
                    lowlink[node] = min(lowlink[node], lowlink[nxt])
                elif nxt in on_stack:
                    # nxt is in the stack, meaning it's on the current SCC search path
                    lowlink[node] = min(lowlink[node], indexes[nxt])

            # If the current node is the root of a strongly connected component
            if lowlink[node] == indexes[node]:
                # Continuously pop from the stack until the entire strongly connected component is retrieved
                comp_id = current_scc_count[0]
                current_scc_count[0] += 1

                scc_nodes = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc_id[w] = comp_id
                    scc_nodes.append(w)
                    if w == node:
                        break

        # Perform strongconnect on all nodes
        for n in all_nodes:
            if n not in indexes:
                strongconnect(n)

        # If there is only one strongly connected component, it means all nodes are in the same cycle or are mutually strongly connected
        # This doesn't matter here, the following logic can handle it as well

        # ---------------------------------------------------------------------
        # 3. Merge each strongly connected component into a new supernode
        # ---------------------------------------------------------------------
        comp_count = current_scc_count[0]  # Total number of strongly connected components
        # Create a new array of supernodes
        super_nodes = [Node(data=set()) for _ in range(comp_count)]

        # Merge Mesh/mass and other information of original nodes into their respective supernodes
        for node in all_nodes:
            cid = scc_id[node]
            # Merge data
            if isinstance(node.data, set):
                # print(f'MERGE: {node.data & super_nodes[cid].data}')
                super_nodes[cid].data.update(node.data)
                super_nodes[cid].forces.update(node.forces)
            else:
                tmp = {node.data}
                # print(f'MERGE: {tmp & super_nodes[cid].data}')
                super_nodes[cid].data.add(node.data)
                super_nodes[cid].forces[node.data] = node.force

            super_nodes[cid].mass = sum(mesh.mass for mesh in super_nodes[cid].data)
            super_nodes[cid].static |= node.static
            super_nodes[cid].force = sum(super_nodes[cid].forces.values())

        # ---------------------------------------------------------------------
        # 4. Build directed edges between supernodes
        #   If two nodes in the original graph belong to different strongly connected components, an edge exists between their supernodes
        # ---------------------------------------------------------------------
        for cid in range(comp_count):
            super_nodes[cid].neighbors = set()

        for node in all_nodes:
            cid_u = scc_id[node]
            for nxt in node.neighbors:
                cid_v = scc_id[nxt]
                if cid_u != cid_v:
                    # Add a directed edge on the new supergraph
                    super_nodes[cid_u].neighbors.add(super_nodes[cid_v])

        # ---------------------------------------------------------------------
        # 5. Find all supernodes with an in-degree of 0 and return them
        # ---------------------------------------------------------------------
        in_degree = [0] * comp_count
        for cid in range(comp_count):
            # Iterate over all neighbors
            for nxt in super_nodes[cid].neighbors:
                # nxt is a supernode
                nxt_id = super_nodes.index(nxt)
                in_degree[nxt_id] += 1

        # Return all supernodes with in_degree = 0
        result = []
        for cid in range(comp_count):
            if in_degree[cid] == 0:
                result.append(super_nodes[cid])

        return result

    @classmethod
    def move_all(cls, grid, mesh_map, nodes, di, dj):
        for node in nodes:
            for mesh in (node.data if isinstance(node.data, set) else {node.data}):
                mesh.remove(grid, mesh_map)
                mesh.pos = mesh.pos[0] + di, mesh.pos[1] + dj
        for node in nodes:
            for mesh in (node.data if isinstance(node.data, set) else {node.data}):
                mesh.place(grid, mesh_map)

    def __repr__(self):
        return str(self.name)