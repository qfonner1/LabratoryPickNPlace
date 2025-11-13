import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# ---------------- PARAMETERS ----------------
padding = 0.075
obstacles = [
    (-0.8, 0.0, 0.52, 0.2+padding, 0.4+padding, 0.52),
    (0.8, 0.0, 0.52, 0.2+padding, 0.4+padding, 0.52),
    (0.0, 0.0, 0.5, 0.2+padding, 0.15+padding, 0.5),
    (0.0, 0.0, 1.5, 0.2+padding, 0.4+padding, 0.5),
    (0,0.5,1,0.2,0.5,1)
]

rand_area = [-1, 1]
z_limits = [0.5, 2.0]
expand_dis = 0.1
path_res = 0.05
goal_sample_rate = 20
max_iter = 5000
robot_radius = 0.1

# ---------------- NODE ----------------
class Node:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.parent = None

# ---------------- RRT* ----------------
class RRTStar3D:
    def __init__(self, start, goal):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.nodes = [self.start]

    def plan(self):
        for _ in range(max_iter):
            rnd = self.random_node()
            nearest = self.nearest_node(rnd)
            new_node = self.steer(nearest, rnd)
            if new_node and self.collision_free(nearest, new_node):
                self.nodes.append(new_node)
                if self.dist_to_goal(new_node) <= expand_dis:
                    final_node = self.steer(new_node, self.goal)
                    if final_node and self.collision_free(new_node, final_node):
                        self.nodes.append(final_node)
                        return self.generate_path(final_node)
        return None

    def steer(self, from_node, to_node):
        dx, dy, dz = to_node.x - from_node.x, to_node.y - from_node.y, to_node.z - from_node.z
        d = math.sqrt(dx**2 + dy**2 + dz**2)
        if d == 0: return None
        scale = min(expand_dis, d) / d
        new_node = Node(from_node.x + dx*scale, from_node.y + dy*scale, from_node.z + dz*scale)
        new_node.parent = from_node
        return new_node

    def random_node(self):
        if random.randint(0, 100) < goal_sample_rate:
            return self.goal
        x = random.uniform(*rand_area)
        y = random.uniform(*rand_area)
        z = random.uniform(*z_limits)
        return Node(x, y, z)

    def nearest_node(self, rnd):
        dists = [(n.x - rnd.x)**2 + (n.y - rnd.y)**2 + (n.z - rnd.z)**2 for n in self.nodes]
        return self.nodes[np.argmin(dists)]

    def dist_to_goal(self, node):
        dx, dy, dz = node.x - self.goal.x, node.y - self.goal.y, node.z - self.goal.z
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def collision_free(self, n1, n2):
        steps = max(int(math.dist([n1.x,n1.y,n1.z],[n2.x,n2.y,n2.z])/path_res), 1)
        for t in np.linspace(0,1,steps):
            x = n1.x + t*(n2.x - n1.x)
            y = n1.y + t*(n2.y - n1.y)
            z = n1.z + t*(n2.z - n1.z)
            for cx, cy, cz, sx, sy, sz in obstacles:
                if (cx-sx-robot_radius <= x <= cx+sx+robot_radius and
                    cy-sy-robot_radius <= y <= cy+sy+robot_radius and
                    cz-sz-robot_radius <= z <= cz+sz+robot_radius):
                    return False
        return True

    def generate_path(self, node):
        path = []
        while node:
            path.append([node.x, node.y, node.z])
            node = node.parent
        return path[::-1]

# ---------------- SMOOTHING ----------------
def shortcut_smooth(path, iterations=200):
    path = path.copy()
    for _ in range(iterations):
        if len(path) <= 2: break
        i, j = sorted(random.sample(range(len(path)), 2))
        if j <= i+1: continue
        if is_collision_free_between(path[i], path[j]):
            path = path[:i+1] + path[j:]
    return path

def is_collision_free_between(p1, p2):
    steps = max(int(math.dist(p1, p2)/path_res), 1)
    for t in np.linspace(0,1,steps):
        x = p1[0] + t*(p2[0]-p1[0])
        y = p1[1] + t*(p2[1]-p1[1])
        z = p1[2] + t*(p2[2]-p1[2])
        for cx, cy, cz, sx, sy, sz in obstacles:
            if (cx-sx-robot_radius <= x <= cx+sx+robot_radius and
                cy-sy-robot_radius <= y <= cy+sy+robot_radius and
                cz-sz-robot_radius <= z <= cz+sz+robot_radius):
                return False
    return True

def rounded_smooth(path, radius=0.1, steps=5):
    path = np.array(path)
    if len(path) <= 2: return path
    smoothed = [path[0]]
    for i in range(1, len(path)-1):
        p_prev, p_curr, p_next = path[i-1], path[i], path[i+1]
        v1, v2 = p_curr - p_prev, p_next - p_curr
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1<1e-6 or l2<1e-6:
            smoothed.append(p_curr)
            continue
        d1, d2 = min(radius, l1/2), min(radius, l2/2)
        p_start, p_end = p_curr - v1/l1*d1, p_curr + v2/l2*d2
        for t in np.linspace(0,1,steps):
            smoothed.append((1-t)*p_start + t*p_end)
    smoothed.append(path[-1])
    return np.array(smoothed)

def densify_path(path, n_points=200):
    """Resample path to have a uniform number of waypoints."""
    path = np.array(path)
    if len(path) < 2:
        return path.tolist()
    # Compute cumulative distance along path
    dist = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    dist = np.insert(dist, 0, 0)
    uniform_dist = np.linspace(0, dist[-1], n_points)
    dense_path = np.zeros((n_points, 3))
    for i in range(3):
        dense_path[:, i] = np.interp(uniform_dist, dist, path[:, i])
    return dense_path.tolist()


def bspline_smooth(path, smooth_factor=0, num_points=300, degree=3):
    """
    Fit a B-spline to the path and sample a smooth trajectory.
    Handles duplicate points and too-short paths robustly.
    """
    path = np.array(path)
    if len(path) < degree + 1:
        return path.tolist()

    # Remove duplicate consecutive points
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    mask = np.insert(diffs > 1e-8, 0, True)
    path = path[mask]

    if len(path) < degree + 1:
        return path.tolist()

    # Parameterize by cumulative distance
    dist = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    dist = np.insert(dist, 0, 0)

    # Ensure strictly increasing
    if np.any(np.diff(dist) <= 0):
        dist = np.arange(len(path))

    try:
        tck, _ = interpolate.splprep(path.T, u=dist, s=smooth_factor, k=min(degree, len(path)-1))
    except Exception as e:
        print(f"[B-spline warning] Fallback to linear interpolation ({e})")
        return densify_path(path, n_points=num_points)  # fallback if spline fails

    u_fine = np.linspace(0, dist[-1], num_points)
    smooth_path = np.array(interpolate.splev(u_fine, tck)).T
    return smooth_path.tolist()



# ---------------- ROBUST PLANNER ----------------
def path_planner(start, goal, max_retries, show_animation):
    for attempt in range(max_retries):
        print(f"[PathPlanner] Attempt {attempt+1}/{max_retries} ...")
        rrt = RRTStar3D(start, goal)
        path = rrt.plan()

        if path and len(path) > 1:
            # Smoothing and densification
            path = shortcut_smooth(path, iterations=200)
            path = rounded_smooth(path, radius=2, steps=5)
            path = densify_path(path, n_points=200)
            # Optional B-spline smoothing
            path = bspline_smooth(path, smooth_factor=0.05, num_points=300)

            if show_animation:
                path_plot(path, start, goal)
            return path

    # If no valid path found, return a simple straight-line fallback
    print("⚠️ No valid path found, returning straight line fallback")
    fallback = np.linspace(start, goal, num=20).tolist()
    return fallback


# ---------------- PLOTTING ----------------
def path_plot(path, start, goal):
    path = np.array(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for cx, cy, cz, sx, sy, sz in obstacles:
        ax.bar3d(cx-sx, cy-sy, cz-sz, sx*2, sy*2, sz*2, alpha=0.3, color='b')
    ax.plot(path[:,0], path[:,1], path[:,2], '-r', linewidth=2)
    ax.scatter(start[0], start[1], start[2], c='g', s=80)
    ax.scatter(goal[0], goal[1], goal[2], c='m', s=80)
    ax.set_xlim(rand_area); ax.set_ylim(rand_area); ax.set_zlim(z_limits)
    plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    start_list = [-0.9, 0, 1.33]
    end_list   = [0.9, 0, 1.33]

    positions = path_planner(start=start_list, goal=end_list, max_retries=20, show_animation=True)
    print(f"✅ Generated {len(positions)} waypoints.")
