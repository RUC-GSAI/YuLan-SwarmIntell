import pulp
import numpy as np


class Node:
    def __init__(self, data):
        self.data = data  # Mesh或set[Mesh]，这对ILP构造无实质影响
        self.detected = False
        self.neighbors = set()  # 存放子节点: Node
        self.force = 0  # 整数，节点自身施加的力
        self.forces = None if isinstance(data, Mesh) else {}
        self.mass = data.mass if isinstance(data, Mesh) else 0  # 已经取过平方根并向下取整
        self.static = data.static if isinstance(data, Mesh) else False  # 若是“静态”节点，则可以把 mass 设为一个很大的值

    @classmethod
    def pass_forward(cls, heads):
        """
        heads: 入度为0的节点列表
        返回: 一个 set, 表示一次ILP求解中得到的可被推动的所有节点
        """

        # 1) 收集所有节点: 从 heads 出发DFS/BFS遍历即可
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

        # 2) 收集边 (v -> w)
        edges = []
        for v in all_nodes:
            for w in v.neighbors:
                edges.append((v, w))

        # 3) 建立 ILP 问题: 最大化 sum x[v]
        prob = pulp.LpProblem("PassForwardILP", pulp.LpMaximize)

        # 设一个大常数 M, 用于“剩余力”的松弛
        # 视实际问题规模而定
        M = 10000

        # 4) 定义变量
        #    x[v]: 0/1, 表示节点v是否被推动
        #    netForce[v]: >=0, 节点v的合力
        #    leftover[v]: >=0, 节点v剩余的可分配力
        #    f_{v->w}: >=0, v分配给w的力
        x = {}
        netForce = {}
        leftover = {}
        f = {}  # f[(v,w)]

        for v in all_nodes:
            if v.static:
                # 若 v 是静态节点, 强制 x[v] = 0
                x[v] = pulp.LpVariable(f"x_static_{id(v)}", cat=pulp.LpBinary)
            else:
                x[v] = pulp.LpVariable(f"x_{id(v)}", cat=pulp.LpBinary)

            netForce[v] = pulp.LpVariable(f"NF_{id(v)}", lowBound=0)
            leftover[v] = pulp.LpVariable(f"LF_{id(v)}", lowBound=0)

        for (v, w) in edges:
            f[(v,w)] = pulp.LpVariable(f"f_{id(v)}_{id(w)}", lowBound=0)

        # 5) 目标函数: 最大化 sum x[v]
        prob += pulp.lpSum(x[v] for v in all_nodes)

        # 6) 添加约束

        # 6.1 静态节点强制 x[v] = 0
        for v in all_nodes:
            if v.static:
                prob.addConstraint( x[v] == 0, f"static_{id(v)}" )

        # 6.2 向下封闭: 若 v被选中 => w也被选中 (v->w)
        #     x[w] >= x[v]
        for (v, w) in edges:
            prob.addConstraint( x[w] >= x[v], f"desc_closure_{id(v)}_{id(w)}" )

        # 6.3 netForce[v] = v.force + sum_{u->v} f[u->v]
        #    先建 in_neighbors[v]
        in_neighbors = { nd: [] for nd in all_nodes }
        for (u, v) in edges:
            in_neighbors[v].append(u)

        for v in all_nodes:
            prob.addConstraint(
                netForce[v] == v.force + pulp.lpSum(f[(u,v)] for u in in_neighbors[v]),
                f"netForceDef_{id(v)}"
            )

        # 6.4 若 x[v]=1 => netForce[v] >= mass[v]
        for v in all_nodes:
            prob.addConstraint(
                netForce[v] >= v.mass * x[v],
                f"minForce_{id(v)}"
            )

        # 6.5 leftover[v] = max(netForce[v]-mass[v], 0)
        #    用不等式 + 大M技巧:
        #    leftover[v] >= netForce[v] - mass[v]
        #    leftover[v] <= netForce[v] - mass[v] + M*(1 - x[v])
        #    leftover[v] >= 0   (已在变量定义)
        for v in all_nodes:
            prob.addConstraint(
                leftover[v] >= netForce[v] - v.mass,
                f"leftover_lb_{id(v)}"
            )
            prob.addConstraint(
                leftover[v] <= netForce[v] - v.mass + M*(1 - x[v]),
                f"leftover_ub_{id(v)}"
            )

        # 6.6 分配给子节点的力 <= leftover[v]
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

        # 6.7 边上的力必须 0, 当子节点不被选
        #     f[v->w] <= M * x[w]
        for (v, w) in edges:
            prob.addConstraint(
                f[(v,w)] <= M * x[w],
                f"edge_flow_{id(v)}_{id(w)}"
            )

        # 7) 求解
        solver = pulp.PULP_CBC_CMD(msg=0)
        result = prob.solve(solver)

        # for var in prob.variables():
        #     print(f'{var.getName()} = {pulp.value(var)}')

        # 8) 若可行(或最优)则收集 x[v]=1 的节点
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
                if mesh_map[i, j] is not None:
                    mesh_map[i, j].node = Node(mesh_map[i, j])
        for i in range(mesh_map.shape[0]):
            for j in range(mesh_map.shape[1]):
                if mesh_map[i, j] is not None:
                    if mesh_map[i, j].node.detected:
                        continue
                    mesh_map[i, j].detect(mesh_map, di, dj, bias)

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
        将 mesh_map 上所有 mesh.node 视为图中的节点，基于 node.neighbors 构造有向图；
        若图中出现环，则对应强连通分量合并为一个新的超级节点，形成一个 DAG。

        返回值:
            返回一个列表，包含所有入度为 0 的“超级节点”(每个超级节点都是一个 Node 对象)，
            这些超级节点中 node.data 可能是一个 set[Mesh]，代表多个 Mesh 合并在一起。
        """

        # ---------------------------------------------------------------------
        # 1. 收集所有节点
        # ---------------------------------------------------------------------
        all_nodes = []
        for i in range(mesh_map.shape[0]):
            for j in range(mesh_map.shape[1]):
                if mesh_map[i][j] is not None:
                    all_nodes.append(mesh_map[i][j].node)

        # 如果没有节点，直接返回
        if not all_nodes:
            return []

        # ---------------------------------------------------------------------
        # 2. 用 Tarjan 算法(或其他算法)找出强连通分量
        # ---------------------------------------------------------------------
        index_counter = [0]  # 记录 DFS 访问顺序
        stack = []
        on_stack = set()
        scc_id = {}  # node -> scc 索引
        indexes = {}  # node -> DFS 访问顺序
        lowlink = {}  # node -> 该节点能追溯到的最小 index
        current_scc_count = [0]  # 已找到的强连通分量数量

        def strongconnect(node):
            # 设置该节点的索引和 lowlink
            indexes[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            # 遍历所有可能的方向
            for nxt in node.neighbors:
                if nxt not in indexes:  # 未访问
                    strongconnect(nxt)
                    lowlink[node] = min(lowlink[node], lowlink[nxt])
                elif nxt in on_stack:
                    # nxt 在栈中，说明在当前强连通分量搜索路径上
                    lowlink[node] = min(lowlink[node], indexes[nxt])

            # 如果当前节点是强连通分量的根
            if lowlink[node] == indexes[node]:
                # 从栈中不断弹出，直到把该强连通分量都取完
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

        # 对所有节点进行 strongconnect
        for n in all_nodes:
            if n not in indexes:
                strongconnect(n)

        # 如果只有一个强连通分量，说明所有节点都在同一个环或互相强连通
        # 这里也无所谓，下面的逻辑同样能处理

        # ---------------------------------------------------------------------
        # 3. 每个强连通分量合并成一个新的超级节点
        # ---------------------------------------------------------------------
        comp_count = current_scc_count[0]  # 强连通分量总数
        # 创建新的超级节点数组
        super_nodes = [Node(data=set()) for _ in range(comp_count)]

        # 将原节点的 Mesh/mass 等信息合并到各自所属的超级节点
        for node in all_nodes:
            cid = scc_id[node]
            # 合并 data
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
        # 4. 建立超级节点之间的有向边
        #   如果原图中两个节点属于不同的强连通分量，则超级节点之间存在边
        # ---------------------------------------------------------------------
        for cid in range(comp_count):
            super_nodes[cid].neighbors = set()

        for node in all_nodes:
            cid_u = scc_id[node]
            for nxt in node.neighbors:
                cid_v = scc_id[nxt]
                if cid_u != cid_v:
                    # 在新的超级图上添加有向边
                    super_nodes[cid_u].neighbors.add(super_nodes[cid_v])

        # ---------------------------------------------------------------------
        # 5. 找出所有入度为 0 的超级节点并返回
        # ---------------------------------------------------------------------
        in_degree = [0] * comp_count
        for cid in range(comp_count):
            # 遍历所有方向
            for nxt in super_nodes[cid].neighbors:
                # nxt 是一个超级节点
                nxt_id = super_nodes.index(nxt)
                in_degree[nxt_id] += 1

        # 返回所有 in_degree = 0 的超级节点
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
