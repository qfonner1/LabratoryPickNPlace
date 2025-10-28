import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d

# =========================================================
#                  PARAMETERS
# =========================================================
# Each obstacle: (cx, cy, cz, sx, sy, sz)
# where c = center, s = half-size (extends in +/- direction)
obstacle_list = [
    (-0.9, 0.0, 0.0, 0.2, 0.4, 0.04),   # left table box
    (0.9, 0.0, 0.0, 0.2, 0.4, 0.04),    # right table box
    (0.0, 0.0, 0.5, 0.2, 0.38, 2.0),     # central robot column
    (0.0,0.5,1.1, 0.2, 0.2, 0.2),
    (0.0,-0.5,1.1, 0.2, 0.2, 0.2),
]

rand_area = [-1, 1]
expand_dis = 0.5
path_resolution = 0.2
goal_sample_rate = 20
max_iter = 2000
robot_radius = 0.2

# =========================================================
#                  NODE & RRT* CLASSES
# =========================================================
class Node:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.parent = None
        self.path_x, self.path_y, self.path_z = [], [], []


class RRTStar3D:
    def __init__(self):
        self.start = None
        self.end = None
        self.node_list = []
        self.min_rand, self.max_rand = rand_area
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.robot_radius = robot_radius
        self.obstacle_list = obstacle_list

    # ---------------- Main Planning ----------------
    def planning(self, start=None, goal=None):
        if start is not None:
            self.start = Node(*start)
        if goal is not None:
            self.end = Node(*goal)
        self.node_list = [self.start]

        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd)
            nearest = self.node_list[nearest_ind]
            new_node = self.steer(nearest, rnd, self.expand_dis)

            if self.is_in_bounds(new_node) and self.check_collision(new_node):
                self.node_list.append(new_node)

            # Try connecting to goal
            if self.dist_to_goal(new_node.x, new_node.y, new_node.z) <= self.expand_dis:
                final = self.steer(new_node, self.end)
                if self.check_collision(final):
                    self.node_list.append(final)
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # No path found

    # ---------------- Core RRT* Functions ----------------
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, from_node.z)
        d, (yaw, pitch) = self.calc_dist_angle(from_node, to_node)
        extend_length = min(extend_length, d)
        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(yaw) * math.cos(pitch)
            new_node.y += self.path_resolution * math.sin(yaw) * math.cos(pitch)
            new_node.z += self.path_resolution * math.sin(pitch)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.path_z.append(new_node.z)

        # ✅ Snap to goal if within one resolution
        if d - extend_length <= self.path_resolution:
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.z = to_node.z
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.path_z.append(to_node.z)

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            return Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(0, 2)
            )
        return Node(self.end.x, self.end.y, self.end.z)

    def get_nearest_node_index(self, rnd):
        dlist = [
            (n.x - rnd.x) ** 2 + (n.y - rnd.y) ** 2 + (n.z - rnd.z) ** 2
            for n in self.node_list
        ]
        return int(np.argmin(dlist))

    def calc_dist_angle(self, from_node, to_node):
        dx, dy, dz = to_node.x - from_node.x, to_node.y - from_node.y, to_node.z - from_node.z
        d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.hypot(dx, dy))
        return d, (yaw, pitch)

    def is_in_bounds(self, node):
        return all(self.min_rand <= v <= self.max_rand for v in (node.x, node.y, node.z))

    # ---------------- Collision Check (Boxes) ----------------
    def check_collision(self, node):
        if node is None or len(node.path_x) == 0:
            return False

        for (cx, cy, cz, sx, sy, sz) in self.obstacle_list:
            for (x, y, z) in zip(node.path_x, node.path_y, node.path_z):
                # Check if point is inside expanded box
                if (cx - sx - self.robot_radius <= x <= cx + sx + self.robot_radius and
                    cy - sy - self.robot_radius <= y <= cy + sy + self.robot_radius and
                    cz - sz - self.robot_radius <= z <= cz + sz + self.robot_radius):
                    return False  # Collision
        return True

    def dist_to_goal(self, x, y, z):
        return math.sqrt((x - self.end.x) ** 2 + (y - self.end.y) ** 2 + (z - self.end.z) ** 2)

    def generate_final_course(self, goal_ind):
        path, node = [], self.node_list[goal_ind]
        while node:
            path.append([node.x, node.y, node.z])
            node = node.parent
        return path[::-1]


# =========================================================
#                  PATH UTILITIES
# =========================================================
def shortcut_smooth(path, check_collision_func, iterations=200):
    path = path.copy()
    for _ in range(iterations):
        if len(path) <= 2:
            break
        i, j = sorted(random.sample(range(len(path)), 2))
        if j <= i + 1:
            continue
        if check_collision_func(path[i], path[j]):
            path = path[:i+1] + path[j:]
    return path

def check_collision_between(p1, p2):
    # linear interpolation and obstacle test
    steps = int(math.dist(p1, p2) / 0.05)
    for (cx, cy, cz, sx, sy, sz) in obstacle_list:
        for t in np.linspace(0, 1, steps):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            z = p1[2] + t * (p2[2] - p1[2])
            if (cx - sx - robot_radius <= x <= cx + sx + robot_radius and
                cy - sy - robot_radius <= y <= cy + sy + robot_radius and
                cz - sz - robot_radius <= z <= cz + sz + robot_radius):
                return False
    return True

def bspline_smooth(path, smooth_factor=0.02):
    path = np.array(path)
    if len(path) < 4:
        # Not enough points for cubic B-spline — use linear interpolation
        t = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, len(path) * 5)
        f = interp1d(t, path, axis=0, kind='linear')
        smooth = f(t_new)
        smooth[0] = path[0]
        smooth[-1] = path[-1]
        return smooth.tolist()

    # Regular B-spline for longer paths
    tck, u = splprep([path[:,0], path[:,1], path[:,2]], s=smooth_factor)
    u_fine = np.linspace(0, 1, len(path) * 5)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    smooth = np.vstack((x_fine, y_fine, z_fine)).T
    smooth[0] = path[0]
    smooth[-1] = path[-1]
    return smooth.tolist()


# =========================================================
#                  VISUALIZATION
# =========================================================
def draw_box(ax, cx, cy, cz, sx, sy, sz, color="b", alpha=0.3):
    """Draw an axis-aligned box centered at (cx,cy,cz)"""
    x = [cx - sx, cx + sx]
    y = [cy - sy, cy + sy]
    z = [cz - sz, cz + sz]

    # 6 faces
    for xi in x:
        Y, Z = np.meshgrid(y, z)
        ax.plot_surface(np.full_like(Y, xi), Y, Z, color=color, alpha=alpha)
    for yi in y:
        X, Z = np.meshgrid(x, z)
        ax.plot_surface(X, np.full_like(X, yi), Z, color=color, alpha=alpha)
    for zi in z:
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, np.full_like(X, zi), color=color, alpha=alpha)


# =========================================================
#                  MAIN PLANNING FUNCTION
# =========================================================
def plan_path(start, goal, show_animation=True):
    rrt_star = RRTStar3D()
    raw_path = rrt_star.planning(start=start, goal=goal)
    if raw_path is None:
        print("⚠️ No path found")
        return []

    simplified = shortcut_smooth(raw_path, check_collision_between, iterations=300)
    smooth = bspline_smooth(simplified)

    if show_animation:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for cx, cy, cz, sx, sy, sz in obstacle_list:
            draw_box(ax, cx, cy, cz, sx, sy, sz, color="b", alpha=0.3)
        raw_np = np.array(raw_path)
        ax.plot(raw_np[:,0], raw_np[:,1], raw_np[:,2], '--k', alpha=0.5, label="Raw RRT* Path")
        smooth_np = np.array(smooth)
        ax.plot(smooth_np[:,0], smooth_np[:,1], smooth_np[:,2], '-r', linewidth=2, label="Smoothed Path")
        ax.scatter(start[0], start[1], start[2], c='g', s=80, label='Start')
        ax.scatter(goal[0], goal[1], goal[2], c='m', s=80, label='Goal')
        ax.set_xlim(rand_area)
        ax.set_ylim(rand_area)
        ax.set_zlim([0,2])
        ax.set_box_aspect([1,1,1])
        plt.legend()
        plt.title("RRT* 3D Path with Shortcut + B-Spline Smoothing")
        plt.show()

    return smooth


# =========================================================
#                  RUN EXAMPLE
# =========================================================
if __name__ == "__main__":
    waypoints = plan_path(start=[-0.9, 0, 1.33], goal=[0.9, 0, 1.33], show_animation=True)
    print(f"✅ Generated {len(waypoints)} waypoints.")
