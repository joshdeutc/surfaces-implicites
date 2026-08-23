"""
Rendu PyVista et export vers les formats d'échange (OBJ / VTK / VTP / PVD).

La fiche d'enregistrement de la TZ demande deux fois PyVista : une fois dans la
prise en main des bibliothèques, une fois pour « l'affichage pyvista (format
vtk) » du réseau temporel. Jusqu'ici tout le projet affichait en matplotlib,
qui suffit pour un contrôle mais pas pour une figure de rapport ni pour
transmettre un maillage à quelqu'un d'autre.

Ce que ce module ajoute par rapport à `visualize.py` :
  - éclairage et lissage corrects (compute_normals + smooth_shading) ;
  - coloration par CHAMP SCALAIRE aux sommets avec barre de couleur légendée,
    indispensable pour montrer une courbure ;
  - caméra REPRODUCTIBLE : deux appels donnent le même cadrage au pixel près ;
  - export vers des formats que ParaView, Blender ou un collègue savent relire.

Tout est hors écran (`pv.OFF_SCREEN = True`) : aucune fenêtre interactive n'est
ouverte, le module est utilisable depuis un script ou un serveur sans affichage.

PIÈGE PYVISTA — le format des faces
-----------------------------------
`skimage.measure.marching_cubes` renvoie des faces (F,3). PyVista attend un
tableau PLAT où chaque triangle est précédé de son nombre de sommets :
[3, i, j, k, 3, i, j, k, ...]. D'où `to_polydata`, qui fait la conversion une
fois pour toutes.

CONVENTION DES SÉRIES TEMPORELLES
---------------------------------
Un fichier .pvd est un simple index XML qui associe un instant à un fichier
.vtp. ParaView le lit comme une animation. On l'écrit à la main : PyVista ne
propose pas d'écrivain .pvd.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv

# Aucune fenêtre : obligatoire avant toute création de Plotter.
pv.OFF_SCREEN = True

CAMERA_DIRECTIONS = {
    "iso": (1.0, 1.0, 0.8),
    "face": (0.0, -1.0, 0.0),
    "dessus": (0.0, 0.0, 1.0),
    "trois-quarts": (1.2, -1.0, 0.55),
}


# ------------------------------------------------------------------
# Construction du maillage
# ------------------------------------------------------------------

def to_polydata(verts, faces, scalars=None, scalar_name="valeur"):
    """
    Maillage PyVista à partir de la sortie de `src.reconstruct.extract_surface`.

    `scalars` (facultatif) : un tableau (V,) attaché AUX SOMMETS, donc
    interpolé continûment sur les triangles au rendu — c'est ce qu'il faut
    pour une courbure, qui est définie point par point et non par facette.
    """
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    faces_pv = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()
    mesh = pv.PolyData(V, faces_pv)
    if scalars is not None:
        s = np.asarray(scalars, dtype=np.float64).ravel()
        if len(s) != len(V):
            raise ValueError(
                f"{scalar_name} a {len(s)} valeurs pour {len(V)} sommets")
        mesh.point_data[scalar_name] = s
        mesh.set_active_scalars(scalar_name)
    return mesh


def prepare(mesh, smooth_iter=0, relaxation=0.05):
    """
    Normales aux sommets (rendu lisse) puis lissage laplacien facultatif.

    Le lissage DÉPLACE les sommets : il embellit mais fausse toute mesure
    géométrique. Il est donc à zéro par défaut, et on ne l'active que pour une
    figure, jamais avant de mesurer une aire, un volume ou une courbure.
    """
    out = mesh.compute_normals(point_normals=True, cell_normals=True,
                               auto_orient_normals=True, consistent_normals=True)
    if smooth_iter > 0:
        active = out.active_scalars_name
        out = out.smooth(n_iter=int(smooth_iter), relaxation_factor=relaxation,
                         feature_smoothing=False, boundary_smoothing=True)
        if active is not None:
            out.set_active_scalars(active)
    return out


# ------------------------------------------------------------------
# Caméra reproductible
# ------------------------------------------------------------------

def camera_for(mesh, direction="iso", distance=2.6, up=(0.0, 0.0, 1.0)):
    """
    Position de caméra déterministe, calculée depuis la boîte englobante.

    On n'utilise PAS le raccourci `camera_position = "iso"` suivi d'un `zoom`,
    dont le résultat dépend de la façon dont VTK ajuste le cadrage. Ici la
    position est explicite : centre de la boîte + une direction fixée, à
    `distance` fois le rayon de la boîte. Deux rendus successifs de la même
    forme, ou de deux formes de tailles différentes, sont donc comparables.
    """
    b = np.asarray(mesh.bounds, dtype=np.float64)
    center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    radius = 0.5 * float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
    radius = max(radius, 1e-9)

    d = np.asarray(CAMERA_DIRECTIONS.get(direction, direction), dtype=np.float64)
    d = d / np.linalg.norm(d)
    position = center + d * radius * float(distance)
    return [tuple(position), tuple(center), tuple(up)]


# ------------------------------------------------------------------
# Rendu
# ------------------------------------------------------------------

def _add(plotter, mesh, scalar_name, cmap, color, clim, show_edges,
         scalar_bar_title, opacity):
    has_scalar = scalar_name is not None and scalar_name in mesh.point_data
    plotter.add_mesh(
        mesh,
        scalars=scalar_name if has_scalar else None,
        cmap=cmap if has_scalar else None,
        clim=clim if has_scalar else None,
        color=None if has_scalar else color,
        show_scalar_bar=False,
        smooth_shading=True,
        specular=0.35,
        specular_power=20,
        ambient=0.18,
        diffuse=0.85,
        opacity=opacity,
        show_edges=show_edges,
        edge_color="black",
        line_width=0.3,
    )
    if has_scalar:
        plotter.add_scalar_bar(
            title=scalar_bar_title or scalar_name,
            n_labels=5, fmt="%.3g", title_font_size=16, label_font_size=13,
            width=0.62, height=0.075, position_x=0.19, position_y=0.035,
            color="black",
        )


def render(mesh, out_path, scalar_name=None, cmap="coolwarm", color="indianred",
           clim=None, title=None, scalar_bar_title=None, direction="iso",
           distance=2.6, window_size=(1200, 950), background="white",
           show_edges=False, show_axes=True, opacity=1.0):
    """
    Enregistre une image PNG du maillage. Hors écran, sans fenêtre.

    `clim` fixe les bornes de la colorimétrie : à préciser dès qu'on compare
    deux figures, sinon chacune est normalisée sur ses propres extrêmes et les
    couleurs ne veulent plus dire la même chose d'une image à l'autre.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    p = pv.Plotter(off_screen=True, window_size=list(window_size))
    p.set_background(background)
    _add(p, mesh, scalar_name, cmap, color, clim, show_edges,
         scalar_bar_title, opacity)
    if show_axes:
        p.add_axes(xlabel="x", ylabel="y", zlabel="z", line_width=2,
                   color="black")
    if title:
        p.add_text(title, position="upper_edge", font_size=13, color="black")
    p.camera_position = camera_for(mesh, direction=direction, distance=distance)
    p.screenshot(out_path)
    p.close()
    return out_path


def render_comparison(meshes, labels, out_path, scalar_name=None,
                      cmap="coolwarm", color="indianred", clim=None,
                      title=None, scalar_bar_title=None, direction="iso",
                      distance=2.6, cell_size=(760, 720), background="white",
                      show_edges=False):
    """
    Plusieurs maillages côte à côte dans une seule image, caméras LIÉES.

    Les caméras sont liées (`link_views`) : les vues subissent exactement la
    même transformation, donc une différence visible entre deux panneaux est
    une différence de FORME, jamais un effet de cadrage. C'est la seule façon
    honnête de mettre deux reconstructions face à face.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = len(meshes)
    if n != len(labels):
        raise ValueError(f"{n} maillages pour {len(labels)} légendes")

    p = pv.Plotter(off_screen=True, shape=(1, n),
                   window_size=[cell_size[0] * n, cell_size[1]])
    p.set_background(background)
    for i, (m, lab) in enumerate(zip(meshes, labels)):
        p.subplot(0, i)
        _add(p, m, scalar_name, cmap, color, clim, show_edges,
             scalar_bar_title, 1.0)
        p.add_text(lab, position="upper_edge", font_size=11, color="black")
    p.link_views()
    p.camera_position = camera_for(meshes[0], direction=direction,
                                   distance=distance)
    if title:
        p.subplot(0, 0)
        p.add_text(title, position="lower_edge", font_size=10, color="black")
    p.screenshot(out_path)
    p.close()
    return out_path


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def export_obj(verts, faces, path):
    """
    Maillage triangulaire au format OBJ, via trimesh.

    OBJ est le format d'échange universel (Blender, MeshLab, impression 3D).
    Il ne transporte AUCUN champ scalaire : la courbure est perdue. Pour la
    conserver, passer par .vtp.
    """
    import trimesh
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    m = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                        faces=np.asarray(faces, dtype=np.int64), process=False)
    m.export(path)
    return path


def export_vtk(mesh, path):
    """
    Maillage + champs scalaires au format VTK (legacy .vtk) ou VTP (XML .vtp).

    C'est le format nommé dans la fiche. Contrairement à OBJ, il transporte les
    champs attachés aux sommets : le maillage exporté reste coloré par sa
    courbure à la réouverture dans ParaView.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mesh.save(path)
    return path


def export_time_series(meshes, times, directory, basename="serie"):
    """
    Série temporelle ParaView : un .vtp par instant + un index .pvd.

    C'est le format qu'attend le champ f(x,y,z,t) : ParaView lit le .pvd, voit
    six instants, et propose directement le curseur d'animation.
    """
    os.makedirs(directory, exist_ok=True)
    if len(meshes) != len(times):
        raise ValueError(f"{len(meshes)} maillages pour {len(times)} instants")

    root = ET.Element("VTKFile", type="Collection", version="0.1",
                      byte_order="LittleEndian")
    coll = ET.SubElement(root, "Collection")
    written = []
    for i, (m, t) in enumerate(zip(meshes, times)):
        name = f"{basename}_{i:03d}.vtp"
        m.save(os.path.join(directory, name))
        written.append(os.path.join(directory, name))
        ET.SubElement(coll, "DataSet", timestep=f"{float(t):.6f}", group="",
                      part="0", file=name)

    pvd = os.path.join(directory, f"{basename}.pvd")
    ET.ElementTree(root).write(pvd, encoding="utf-8", xml_declaration=True)
    return pvd, written


def relire(path):
    """
    Relit un fichier exporté et renvoie (nb_sommets, nb_faces, champs).

    Sert de PREUVE que l'export est valide : un fichier de taille non nulle
    peut très bien être illisible ou vide de géométrie. On ne se fie donc pas à
    l'octet, on relit et on recompte.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        import trimesh
        m = trimesh.load(path, process=False, force="mesh")
        return len(m.vertices), len(m.faces), []
    m = pv.read(path)
    n_faces = int(m.n_cells)
    return int(m.n_points), n_faces, list(m.point_data.keys())


def taille_ko(path):
    return os.path.getsize(path) / 1024.0
