import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def size_gradient(d1, d2, x, y):
    return (d2 - d1) / 10 * y + d1


def Heaviside(x: np.ndarray):
    indicator = np.array(x > 0, dtype=float)
    return x * indicator


def compute_velocity(p1, p2, k: float):
    d_1 = size_gradient(d1, d2, p1[:, 0], p1[:, 1])
    d_2 = size_gradient(d1, d2, p2[:, 0], p2[:, 1])
    l = np.linalg.norm(p2 - p1, axis=1).reshape(-1, 1)
    unit = (p2 - p1) / l
    return k * Heaviside(np.asarray(0.5 * (d_1 + d_2).reshape(l.shape) - l)) * unit


def check_collision(p, index) -> bool:
    x, y = index[0, 0], index[0, 1]
    collision = False
    for i in range(max(0, x - 2), min(res_x, x + 3)):
        for j in range(max(0, y - 2), min(res_y, y + 3)):
            if grid[i, j] != -1:
                q = coords[grid[i, j], :]
                if np.linalg.norm(q - p) < radius - 1e-6:
                    collision = True
    return collision


def escape(fixed_seeds: list, p2: np.ndarray, k: float, dt: float):
    if len(fixed_seeds) > 1:
        seeds = np.array(fixed_seeds).squeeze()
    else:
        seeds = np.array(fixed_seeds).reshape(1, -1)

    vs = compute_velocity(seeds, p2, k)
    v = np.sum(vs, axis=0)
    if np.linalg.norm(v) > 1e-4:
        print("begin to escape")
    while np.linalg.norm(v) > 1e-4:
        plt.cla()
        p2 += v * dt
        vs = compute_velocity(seeds, p2, k)
        v = np.sum(vs, axis=0)

    print("finish escaping")


if __name__ == "__main__":
    # 10*10 region
    d1 = 1
    d2 = 5
    dt = 0.1
    k = 0.1

    coords = np.zeros((5, 2))
    fixed_seeds = []
    radius = 0.5
    dx = radius / np.sqrt(2)
    res_x = res_y = int(10 / dx) + 1
    res = (res_x, res_y)
    grid = np.zeros((res_x, res_y), dtype=int) - 1
    coords[0, :] = 0.5, 0.5
    fixed_seeds.append(np.array([[0.5, 0.5]]))
    grid[int(0.5 / dx), int(0.5 / dx)] = 0
    n = 1
    while n < 5:
        coord = np.random.rand(1, 2) * 10
        dg = size_gradient(d1=d1, d2=d2, x=coord[0, 0], y=coord[0, 1])
        alpha = np.random.uniform()
        index = np.array(coord / dx, dtype=int)
        # print(index)
        collision = check_collision(coord, index)
        if not collision:
            if (d1 / dg) ** 2 > alpha:
                escape(fixed_seeds, coord, k, dt)
                fixed_seeds.append(coord)
                coords[n, :] = coord
                grid[index] = n
                n += 1

    plt.scatter(coords[:, 0], coords[:, 1])
    plt.show()

