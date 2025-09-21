import numpy as np

def intersecting(p, qs, edges, num_batches=8, tol=1e-8):
    # sort edges based on distance to p
    sorted_idcs = np.argsort(
        (edges[:, 0, 0] - p[0])**2 + (edges[:, 0, 1] - p[1])**2
    )
    edges = edges[sorted_idcs]

    # batchsize increases exponentially
    batch_bounds = np.unique(
        np.geomspace(1, len(edges), num=num_batches + 1, dtype=int)
    )
    remaining = np.ones(len(qs), dtype=bool)
    
    v = p - qs
    w = edges[:, 1] - edges[:, 0]
    b = p - edges[:, 0]
    
    for start, end in zip(batch_bounds[:-1], batch_bounds[1:]):
        # only check qs that werent intersected yet
        v1, v2 = v[remaining, 0], v[remaining, 1]              # shape n x 2
        w1, w2 = w[start:end, 0], w[start:end, 1]              # shape m x 2
        b1, b2 = b[start:end, 0], b[start:end, 1]              # shape m x 2
        
        x1detW = w2 * b1 - w1 * b2                             # shape m
        x2detW = np.array([-v2, v1]).T @ [b1, b2]              # shape n x m

        # det of 2x2 is v1 w2 - v2 w1
        detW = np.outer(v1, w2) - np.outer(v2, w1)             # shape n x m
        sgn = np.sign(detW)                                    # shape n x m
       
        # invert values with detW < 0
        y1 = sgn * x1detW                                      # shape n x m
        y2 = sgn * x2detW                                      # shape n x m
        
        # for every q and every edge save if they intersect
        bound = (1 - tol) * np.abs(detW)                       # shape n x m
        intersects = (                                         # shape n x m
            (tol < y1) & (y1 < bound) &
            (tol < y2) & (y2 < bound)
        )

        remaining[remaining] = ~np.any(intersects, axis=1)     # shape n 

    return ~remaining



def check_path(path, mid, borders, obstacles, obstacle_radius=0.5):
    for p, q in zip(path[:-1], path[1:]):
        r, _ = reachable(p, q[None], np.vstack((borders, obstacles)), threshold=obstacle_radius**2)
        
        if not r[0]:
            inside, _ = reachable(mid, path[-1][None], borders, threshold=obstacle_radius**2)

            if not inside[0]:
                print('Path blocked - End point outside arena')
                return False, False

            print('Path blocked - End point inside arena')
            return False, True

    return True, True

def cone_check(p, nodes, edges, cone_tol=1e-8):
    # create cone of two edges
    neigh1_idx = np.where(np.all(edges[:, 0] == p, axis=1))[0][0]
    neigh2_idx = np.where(np.all(edges[:, 1] == p, axis=1))[0][0]
    
    neighbors = [edges[neigh1_idx, 1], edges[neigh2_idx, 0]]
    cone = (neighbors - p).T

    # if det(cone) == 0 no points are within it
    if np.linalg.det(cone) == 0:
        return np.zeros(len(nodes), dtype=bool)

    connect = (nodes - p).T
    lin_comb = np.linalg.solve(cone, connect)

    return (
        (cone_tol < lin_comb[0]) & 
        (cone_tol < lin_comb[1])
    )

def reachable(p, qs, obstacles, threshold=0.0144):
    v = qs - p                                              # shape n x 2
    w = p - obstacles                                       # shape m x 2

    vTv = (v[:, 0]**2 + v[:, 1]**2)[:, None] +1e-8          # shape n x 1
    vTw = v @ w.T                                           # shape n x m
    
    ts = np.clip(-vTw / vTv, 0, 1)                          # shape n x m   
    
    con_x = p[0] + ts * v[:, 0][:, None] - obstacles[:, 0]  # shape n x m
    con_y = p[1] + ts * v[:, 1][:, None] - obstacles[:, 1]  # shape n x m
    
    dists = con_x**2 + con_y**2                             # shape n x m
    min_dists = np.min(dists, axis=1)                       # shape n

    mask = min_dists > threshold                            # shape n
    return mask, min_dists[mask]

def find_path(start, end, nodes, obstacles, obstacle_radius=0.25, bias=0.2, avoidance=0):
    # add start and end to nodes
    nodes = np.vstack(([start], [end], nodes))
    
    n = len(nodes)
    
    G = np.full(n, np.inf)
    H = np.linalg.norm(nodes - end, axis=1)
    # mask explored nodes
    F = np.ma.MaskedArray(
        np.full(n, np.inf), 
        np.zeros(n, dtype=bool)
    )
    pred = np.full(n, -1, dtype=int)

    curr_idx = 0
    G[curr_idx] = 0
    
    while curr_idx != 1:
        # if all values in F are inf argmin will return maksed index
        if F.mask[curr_idx] == True:
            # changes goal node to closest node to end point
            dists = np.linalg.norm(nodes[pred != -1] - end, axis=1)
            # if final point is unreachable
            if len(dists) == 0:
                curr_idx = 1 + np.argmin(np.linalg.norm(nodes[1:] - start, axis=1))
                G[curr_idx] = 0
                print('Start node inside obstacle, instead starting from closest node')
            # else start point is inside obstacle
            else:
                curr_idx = np.arange(len(nodes))[pred != -1][np.argmin(dists)]
                print('No valid path found, instead pathing to closest node')
                break
                
        current = nodes[curr_idx]
        
        # use helper functions to obtain indices of visable nodes
        visable, min_dists = reachable(current, nodes, obstacles, threshold=obstacle_radius**2)
        idcs = np.nonzero(visable)[0]
        
        # update predecessor array and G based on distances from current to next points
        new_Gs = G[curr_idx] + np.linalg.norm(nodes[idcs] - current, axis=1) + bias + avoidance * np.exp(min_dists - obstacle_radius**2)
        improved = new_Gs < G[idcs]
        
        G[idcs[improved]] = new_Gs[improved]
        pred[idcs[improved]] = curr_idx

        # greed = 0 is A* wich finds the best path, greed = 1 is greedy search
        F.data[idcs] = G[idcs] + H[idcs]
        F.mask[curr_idx] = True
        curr_idx = np.argmin(F)
    
    # backtracking predecessor array
    path = [nodes[curr_idx]]
    while pred[curr_idx] != -1:
        curr_idx = pred[curr_idx]
        path.append(nodes[curr_idx])
    
    return np.array(path[::-1])


def create_map(borders, obstacles, center=np.array([-1, 0]), nodes_per_obstacle=10, min_radius=0.5):
    # sample nodes around borders facing center
    to_center = center - borders
    phi = np.arctan2(to_center[:, 1], to_center[:, 0])
    
    phis = np.linspace(phi - np.pi / 2, phi + np.pi / 2, nodes_per_obstacle // 2).T

    rs = min_radius + 0.2 * np.exp(np.random.normal(0, 0.5, size=(len(borders), nodes_per_obstacle//2)))
    
    borders_x = borders[:, 0, None] + rs * np.cos(phis)
    borders_y = borders[:, 1, None] + rs * np.sin(phis)

    border_points = np.stack((borders_x, borders_y), axis=-1).reshape(-1, 2)
    
    # sample nodes around obstacles
    phis = np.linspace(0, 2 * np.pi, nodes_per_obstacle, endpoint=False)

    rs = min_radius + 0.2 * np.exp(np.random.normal(0, 0.5, size=(len(obstacles), nodes_per_obstacle)))
    
    obstacles_x = obstacles[:, 0, None] + rs * np.cos(phis)
    obstacles_y = obstacles[:, 1, None] + rs * np.sin(phis)
    
    obstacle_points = np.stack((obstacles_x, obstacles_y), axis=-1).reshape(-1, 2)

    nodes = np.vstack((border_points, obstacle_points))
    # filter out nodes that are inside obstacles
    con = nodes[:, None] - np.vstack((borders, obstacles))[None]
    dists = con[..., 0]**2 + con[..., 1]**2
    min_dists = np.min(dists, axis=1)

    #return nodes
    return nodes[min_dists > min_radius**2]


def sample_centric_points(x,y,radius=0.1,n_points=5):
    phi = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    return np.c_[x + radius * np.cos(phi), y + radius * np.sin(phi)]


def combine_path(start, end, borders, obstacles, obstacle_radius=0.12, bias=0.1):
    nodes = create_map(borders, obstacles, min_radius=obstacle_radius)
    path = find_path(start, end, nodes, np.vstack((borders, obstacles)), obstacle_radius=obstacle_radius, bias=bias)

    return path



"""
class Explorer:
    def __init__(self, num_cells, grid_width, grid_offset=np.array((-1, 0))):
        self.grid = np.zeros((num_cells, num_cells))
        self.width = grid_width
        self.offset = grid_offset

    def update(self, robot_pos, std=0.2):
        x_pos, y_pos = robot_pos - self.offset
        bound = self.width / 2
        
        x, y = np.meshgrid(
            np.linspace(-bound, bound, num=self.grid.shape[0]),
            np.linspace(-bound, bound, num=self.grid.shape[1])
        )
        self.grid += np.exp(-((x - x_pos)**2 + (y - y_pos)**2) / std)
        self.grid = (self.grid - self.grid.mean()) / self.grid.std()

    def get_point(self, robot_pos, borders, obstacles, num_points=5, obstacle_radius=0.12, e=1, d=1, a=5):
        # sample points
        n, m = self.grid.shape
        bound = self.width / 2

        points = np.random.uniform(-bound, bound, size=(num_points, 2))
        
        # filter out points inside obstacles
        obstacle_dists = np.ones(len(points)) * 100
        if len(obstacles) > 0:
            con = points[:, None] - (obstacles - self.offset)[None]
            dists = con[..., 0]**2 + con[..., 1]**2
            obstacle_dists = np.min(dists, axis=1)

        points = points[obstacle_dists > obstacle_radius**2]
        
        # filter out points outside arena
        mask, _ = reachable([0, 0], points, (borders - self.offset), threshold=obstacle_radius**2)
        points = points[mask]

        x_idcs = ((points[:, 0] + bound) * (n - 1) / self.width).astype(int)
        y_idcs = ((points[:, 1] + bound) * (m - 1) / self.width).astype(int)

        # explo loss avoid already explored regions
        expl_loss = e * self.grid[y_idcs, x_idcs]

        # dist loss choses close points
        con = points - (robot_pos - self.offset)
        dist_loss = d * np.linalg.norm(con, axis=1)

        # avoid loss choses points far away from obstacles and borders
        con = points[:, None] - (borders - self.offset)[None]
        dists = con[..., 0]**2 + con[..., 1]**2
        border_dists = np.min(dists, axis=1)

        min_dists = np.minimum(obstacle_dists[obstacle_dists > obstacle_radius**2][mask], border_dists)

        avoid_loss = a * np.exp(-(min_dists - obstacle_radius**2) / 0.1)

        # sum losses
        loss = expl_loss + dist_loss + avoid_loss
        
        return points[np.argmin(loss)] + self.offset

        #return points[:, 0], points[:, 1], loss
"""


class Explorer:
    def __init__(self,num_cells,grid_width,grid_offset=np.array([-1,0])):
        
        self.width = grid_width
        self.offset = grid_offset
        self.grid = np.zeros((num_cells, num_cells))
        
        bound = self.width / 2
        xs = np.linspace(-bound, bound, num_cells)
        ys = np.linspace(-bound, bound, num_cells)
        self.X, self.Y = np.meshgrid(xs, ys)

    def update(self, robot_pos, sigma=0.2, decay = 0.99):
        rel_pos  =robot_pos - self.offset
        gauss = np.exp(-(((self.X - rel_pos[0])**2 + (self.Y - rel_pos[1])**2) / 2*sigma**2))       # macht mal sigma**2
        self.grid = decay * self.grid + gauss
        self.grid = np.clip(self.grid, 0, 1e3)

    def get_point(self,robot_pos,borders, obstacles, num_candidates = 50, obstacle_radius=0.12,weight_expl=1.0,weight_dist=1.0,weight_avoid=5.0):

        n,m = self.grid.shape
        bound = self.width/2
        candidates = np.random.uniform(-bound,bound,size=(num_candidates, 2))

        if len(obstacles) > 0:
            con = candidates[:, None] - (obstacles - self.offset)[None]
            dists = np.sum(con**2, axis=2)
            obs_min_dists = np.min(dists, axis=1)
            
        else:
            obs_min_dists = np.full(num_candidates, np.inf)

        valid = obs_min_dists > obstacle_radius **2      # muss nicht quadriert werden

        if len(borders) > 0:
            reachable_mask, _ = reachable([0, 0], candidates, (borders - self.offset), threshold=obstacle_radius**2)
            valid = valid & reachable_mask

        if not np.any(valid):
            return robot_pos

        candidates = candidates[valid]
        x_idx = ((candidates[:, 0] + bound) * (n - 1) / self.width).astype(int)
        y_idx = ((candidates[:, 1] + bound) * (m - 1) / self.width).astype(int)

        expl_loss = weight_expl * self.grid[y_idx, x_idx]
        dist_loss = weight_dist * np.linalg.norm(candidates - (robot_pos - self.offset), axis=1)


        if len(borders) > 0:
            con_border = candidates[:, None] - (borders - self.offset)[None]
            border_dists = np.min(np.sum(con_border**2, axis=2), axis=1)
        else:
            border_dists = np.full(len(candidates), np.inf)
        min_dists = np.minimum(obs_min_dists[valid][:len(candidates)], border_dists)
        
        avoid_loss = weight_avoid * np.exp(-(min_dists - obstacle_radius**2) / 0.1)

        total_loss = expl_loss + dist_loss + avoid_loss

        best_candidate = candidates[np.argmin(total_loss)]
        return best_candidate + self.offset


def end_check(borders, mid=[-1, 0], obstacle_radius = 0.2):
    r = 8
    phis = np.linspace(0, 2 * np.pi, num = 50, endpoint=False)
    xs = r * np.cos(phis)
    ys = r * np.sin(phis)

    circle_points = np.column_stack([xs, ys]) + mid
    reach = reachable(mid, circle_points, borders, threshold=obstacle_radius**2)[0]
    print(f"This is the number of reachable points still : {np.sum(reach)}")
    return not np.any(reach)
