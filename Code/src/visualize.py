"""
Section 7 du notebook : visualisation 3D.

Différence avec le notebook : celui-ci utilise `%matplotlib ipympl`
(affichage interactif). En script, on utilise le backend "Agg" qui
sauvegarde la figure dans un PNG (pas d'écran requis).
"""
import matplotlib
matplotlib.use("Agg")  # backend non-interactif -> on enregistre un fichier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_surface(verts, faces, surface_points, box, out_path="output/surface.png"):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.35)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    # Points de surface utilisés pour l'apprentissage (1 sur 20)
    ax.scatter(
        surface_points[::20, 0],
        surface_points[::20, 1],
        surface_points[::20, 2],
        s=4,
        alpha=0.6,
    )

    ax.set_xlim(-box, box)
    ax.set_ylim(-box, box)
    ax.set_zlim(-box, box)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Surface implicite apprise par réseau de neurones")

    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Figure sauvegardée -> {out_path}")
