import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
from saving_config import BASE_OUTPUT_DIR
import saving_config
import plotly.graph_objects as go


# ---------------- PARAMETERS ----------------
padding = 0.075
obstacles = [
    (-0.8, 0.0, 0.52, 0.2+padding, 0.4+padding, 0.52),
    (0.8, 0.0, 0.52, 0.2+padding, 0.4+padding, 0.52),
    (0.0, 0.0, 0.5, 0.2+padding, 0.15+padding, 0.5),
    (0.0, 0.0, 1.5, 0.2+padding, 0.4+padding, 0.5),
    (0,1.0,1,0.2,0.5,1),
    (0,-1.0,0.6,0.2,0.5,0.6),
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
        print(f"[Path Planner] Fallback to linear interpolation ({e})")
        return densify_path(path, n_points=num_points)  # fallback if spline fails

    u_fine = np.linspace(0, dist[-1], num_points)
    smooth_path = np.array(interpolate.splev(u_fine, tck)).T
    return smooth_path.tolist()



# ---------------- ROBUST PLANNER ----------------
def path_planner(start, goal, max_retries, show_animation):
    for attempt in range(max_retries):
        print(f"[Path Planner] Attempt {attempt+1}/{max_retries} ...")
        rrt = RRTStar3D(start, goal)
        rrt_path = rrt.plan()

        if rrt_path and len(rrt_path) > 1:
            shortcut_path = shortcut_smooth(rrt_path, iterations=200)
            rounded_path = rounded_smooth(shortcut_path, radius=2, steps=5)
            densified_path = densify_path(rounded_path, n_points=200)
            final_path = bspline_smooth(densified_path, smooth_factor=0.05, num_points=300)


            path_plot(rrt_path, shortcut_path, rounded_path, final_path, start, goal, show_animation)

            return final_path

    print("[Path Planner] No valid path found, returning straight line fallback")
    fallback = np.linspace(start, goal, num=20).tolist()
    return fallback, fallback, fallback


# ---------------- PLOTTING ----------------
def path_plot(rrt_path, shortcut_path, rounded_path, final_path, start, goal, show_animation=False, save_html=True):
    # --- Matplotlib plot (unchanged) ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot obstacles
    for cx, cy, cz, sx, sy, sz in obstacles:
        ax.bar3d(cx-sx, cy-sy, cz-sz, sx*2, sy*2, sz*2, alpha=0.3, color='w')

    # Plot paths
    rrt_path = np.array(rrt_path)
    shortcut_path = np.array(shortcut_path)
    rounded_path = np.array(rounded_path)
    final_path = np.array(final_path)

    ax.plot(rrt_path[:,0], rrt_path[:,1], rrt_path[:,2], '-k', linewidth=1, label="RRT* path")
    ax.plot(shortcut_path[:,0], shortcut_path[:,1], shortcut_path[:,2], '-b', linewidth=2, label="Shortcut path")
    ax.plot(rounded_path[:,0], rounded_path[:,1], rounded_path[:,2], '-y', linewidth=2, label="Rounded path")
    ax.plot(final_path[:,0], final_path[:,1], final_path[:,2], '-r', linewidth=3, label="Final Smoothed path")

    # Start and goal
    ax.scatter(start[0], start[1], start[2], c='g', s=80, label="Start")
    ax.scatter(goal[0], goal[1], goal[2], c='r', s=80, label="Goal")

    # Waypoints at 1/3 and 2/3 along the final path
    dist_start_goal = math.dist(start, goal)
    if dist_start_goal > 0.1:
        n = len(final_path)
        idx1, idx2 = n//3, 2*n//3
        ax.scatter(final_path[idx1,0], final_path[idx1,1], final_path[idx1,2], c='c', s=60, label='_nolegend_')
        ax.scatter(final_path[idx2,0], final_path[idx2,1], final_path[idx2,2], c='c', s=60, label='_nolegend_')

    # title, labels, limits, legend
    ax.set_title("End Effector Path")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(rand_area); ax.set_ylim(rand_area); ax.set_zlim(z_limits)
    ax.legend()
    if show_animation is True:
        plt.show()

    # Save PNG
    saving_config.PLOT_COUNTER += 1
    plot_filename = os.path.join(BASE_OUTPUT_DIR, f"path_{saving_config.PLOT_COUNTER}.png")
    fig.savefig(plot_filename)
    print(f"[Path Planner] Plot saved to: {plot_filename}")
    plt.close(fig)

    # Save interactive HTML with Plotly ---
    if save_html:
        fig_html = go.Figure()

        # Obstacles as semi-transparent cubes
        for cx, cy, cz, sx, sy, sz in obstacles:
            # Six faces of the cube

            # XY planes (top and bottom)
            for z0 in [cz - sz, cz + sz]:
                fig_html.add_trace(go.Surface(
                    x=[[cx-sx, cx+sx], [cx-sx, cx+sx]],
                    y=[[cy-sy, cy-sy], [cy+sy, cy+sy]],
                    z=[[z0, z0], [z0, z0]],
                    colorscale=[[0, 'white'], [1, 'white']],
                    opacity=1.0,
                    showscale=False,
                    name='Obstacle',
                ))

            # XZ planes (front and back)
            for y0 in [cy - sy, cy + sy]:
                fig_html.add_trace(go.Surface(
                    x=[[cx-sx, cx+sx], [cx-sx, cx+sx]],
                    y=[[y0, y0], [y0, y0]],
                    z=[[cz-sz, cz-sz], [cz+sz, cz+sz]],
                    colorscale=[[0, 'white'], [1, 'white']],
                    opacity=1.0,
                    showscale=False,
                    name='Obstacle',
                ))

            # YZ planes (left and right)
            for x0 in [cx - sx, cx + sx]:
                fig_html.add_trace(go.Surface(
                    x=[[x0, x0], [x0, x0]],
                    y=[[cy-sy, cy-sy], [cy+sy, cy+sy]],
                    z=[[cz-sz, cz+sz], [cz-sz, cz+sz]],
                    colorscale=[[0, 'white'], [1, 'white']],
                    opacity=1.0,
                    showscale=False,
                    name='Obstacle',
                ))




        # Paths
        fig_html.add_trace(go.Scatter3d(x=rrt_path[:,0], y=rrt_path[:,1], z=rrt_path[:,2],
                                        mode='lines', line=dict(color='black', width=4), name='RRT* path'))
        fig_html.add_trace(go.Scatter3d(x=shortcut_path[:,0], y=shortcut_path[:,1], z=shortcut_path[:,2],
                                        mode='lines', line=dict(color='blue', width=6), name='Shortcut path'))
        fig_html.add_trace(go.Scatter3d(x=final_path[:,0], y=final_path[:,1], z=final_path[:,2],
                                        mode='lines', line=dict(color='red', width=8), name='Final path'))

        # Start and goal
        fig_html.add_trace(go.Scatter3d(x=[start[0]], y=[start[1]], z=[start[2]],
                                        mode='markers', marker=dict(size=6, color='green'), name='Start'))
        fig_html.add_trace(go.Scatter3d(x=[goal[0]], y=[goal[1]], z=[goal[2]],
                                        mode='markers', marker=dict(size=6, color='red'), name='Goal'))

        # Waypoints
        if dist_start_goal > 0.1:
            fig_html.add_trace(go.Scatter3d(x=[final_path[idx1,0], final_path[idx2,0]],
                                            y=[final_path[idx1,1], final_path[idx2,1]],
                                            z=[final_path[idx1,2], final_path[idx2,2]],
                                            mode='markers', marker=dict(size=6, color='cyan'),
                                            name='Waypoints'))


        # Layout
        fig_html.update_layout(scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(range=rand_area),
            yaxis=dict(range=rand_area),
            zaxis=dict(range=z_limits),
            aspectmode='cube'  # ensures equal scaling
        ))
        html_filename = os.path.join(BASE_OUTPUT_DIR, f"path_{saving_config.PLOT_COUNTER}.html")
        fig_html.write_html(html_filename)
        print(f"[Path Planner] Interactive HTML plot saved to: {html_filename}")




# ---------------- MAIN ----------------
# if __name__ == "__main__":
#     start_list = [-0.9, 0, 1.33]
#     end_list   = [0.9, 0, 1.33]

#     final_path = path_planner(start=start_list, goal=end_list, max_retries=20, show_animation=True)
#     print(f"✅ Generated {len(final_path)} waypoints in final path.")
