import numpy as np

BOUNDS = (0.0, 10.0, 0.0, 10.0)  # xmin, xmax, ymin, ymax


def _poly(points):
    return np.array(points, dtype=float)


def harta_3_obstacole():
    """Trei obstacole de forme diferite (triunghi, patrulater, dreptunghi)."""
    return {
        "name": "3 obstacole",
        "start": np.array([1.0, 1.0]), #punct de start
        "stop": np.array([9.0, 9.0]), #punct de sfarsit
        "obstacles": [
            _poly([(2.0, 4.0), (4.0, 4.0), (3.0, 6.5)]),
            _poly([(5.0, 2.0), (7.5, 2.5), (7.0, 4.5), (5.0, 4.0)]),
            _poly([(5.5, 6.5), (8.0, 6.5), (8.0, 8.0), (5.5, 8.0)]),
        ],
        "bounds": BOUNDS,
    }


def harta_5_obstacole():
    return {
        "name": "5 obstacole",
        "start": np.array([1.0, 1.0]),
        "stop": np.array([9.0, 9.0]),
        "obstacles": [
            _poly([(1.5, 3.5), (3.5, 3.5), (2.5, 5.5)]),
            _poly([(4.5, 1.5), (6.5, 2.0), (6.0, 3.8), (4.2, 3.3)]),
            _poly([(7.5, 3.5), (9.0, 3.5), (9.0, 5.0), (7.5, 5.0)]),
            _poly([(4.0, 5.5), (5.5, 5.2), (6.3, 6.5), (5.2, 7.5), (3.8, 7.0)]),
            _poly([(7.0, 6.3), (8.5, 6.3), (8.5, 7.8), (7.0, 7.8)]),
        ],
        "bounds": BOUNDS,
    }


def harta_U():

    return {
        "name": "U",
        "start": np.array([1.0, 5.0]),
        "stop": np.array([9.0, 5.0]),
        "obstacles": [
            _poly([
                (4.0, 2.0), (7.0, 2.0), (7.0, 8.0), (4.0, 8.0),
                (4.0, 7.0), (6.0, 7.0), (6.0, 3.0), (4.0, 3.0),
            ]),
        ],
        "bounds": BOUNDS,
    }


MAPS = {
    "3": harta_3_obstacole,
    "5": harta_5_obstacole,
    "U": harta_U,
}


def get_map(name):
    #Intoarce harta ceruta: '3', '5' sau 'U'.
    return MAPS[name]()


def plot_map(ax, m):
    from matplotlib.patches import Polygon

    xmin, xmax, ymin, ymax = m["bounds"]
    for obs in m["obstacles"]:
        ax.add_patch(Polygon(obs, closed=True, facecolor="#4a90d9",
                             edgecolor="black", alpha=0.8))
    ax.plot(*m["start"], "go", markersize=10, label="START")
    ax.plot(*m["stop"], "r*", markersize=15, label="STOP")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Harta: " + m["name"])
    ax.legend(loc="upper left")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, MAPS):
        plot_map(ax, get_map(name))
    plt.tight_layout()
    plt.show()