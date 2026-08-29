"""
Surfaces complexes : le stress-test TOPOLOGIQUE du projet.

Tout ce que le projet a validé jusqu'ici — sphère, ellipsoïde, globule rouge,
séquence temporelle — est de GENRE 0 : des ballons sans trou. L'énoncé du
sujet s'ouvre pourtant sur l'argument que les équations implicites savent
représenter des surfaces « très complexes en termes de forme ET DE
TOPOLOGIE », et la fiche d'enregistrement consacre une ligne aux surfaces de
Chmutov, gyroïdes, metaballs et surfaces de Barth. C'est ici qu'on vérifie.

Deux questions, deux parties :

  A. Ces surfaces sont-elles seulement compatibles avec le pipeline ?
     On MESURE la topologie du maillage (chi, genre, étanchéité) au lieu de la
     supposer. Le résultat n'est pas celui qu'on attendait : deux des quatre
     surfaces demandées ne sont pas fermées, et le pipeline suppose partout une
     surface fermée. C'est une limite de la méthode, pas un bug — on la
     documente au lieu de la contourner.

  B. Le réseau les apprend-il ? Chamfer sur chacune, avec ou sans le terme de
     signe selon que « dedans » a un sens.

Lancer :  /usr/local/bin/python3 main_shapes.py
"""
import os
import time

import numpy as np
import torch
from skimage import measure

from src.shapes_complexes import catalogue, sample_surface
from src.invariants import euler_characteristic, is_watertight, mesh_area
from src.model import IGRNet
from src.train import train_igr
from src.reconstruct import evaluate_grid, extract_surface
from src.metrics import chamfer_distance, eikonal_error
from src import viz_pv

RES_TABLE = 128      # grille du tableau topologique
RES_APPRIS = 128     # grille de reconstruction du champ appris
EPOCHS = 4000


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def maille_exacte(f, box, res=RES_TABLE):
    """Marching cubes sur le champ EXACT, et test de découpe par la boîte."""
    ax = np.linspace(-box, box, res)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    V = np.asarray(f(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
                   ).reshape(res, res, res)
    # La surface traverse-t-elle une face de la boîte ? Si oui le maillage est
    # tronqué, et chi mesuré décrit la TRONCATURE, pas la surface.
    faces_boite = [V[0], V[-1], V[:, 0], V[:, -1], V[:, :, 0], V[:, :, -1]]
    coupee = any((float(fb.min()) < 0.0 < float(fb.max())) for fb in faces_boite)
    sp = ax[1] - ax[0]
    verts, faces, _, _ = measure.marching_cubes(V, 0.0, spacing=(sp, sp, sp))
    verts = verts - box
    return verts, faces, coupee


# ==================================================================
# A. Tableau topologique
# ==================================================================

def partie_A():
    print("=" * 96)
    print("A. TOPOLOGIE MESURÉE DES SURFACES COMPLEXES  (grille %d^3, champ exact)"
          % RES_TABLE)
    print("=" * 96)
    print(f"{'surface':<22}{'boîte':>7}{'sommets':>9}{'faces':>9}"
          f"{'chi':>7}{'genre':>8}{'étanche':>9}{'coupée':>9}   remarque")
    print("-" * 96)

    lignes = []
    for s in catalogue():
        v, f, coupee = maille_exacte(s["f"], float(s["box"]))
        topo = euler_characteristic(v, f)
        wt = is_watertight(v, f)
        print(f"{s['nom']:<22}{float(s['box']):>7.3f}{topo['V']:>9d}{topo['F']:>9d}"
              f"{topo['chi']:>7d}{topo['genus']:>8.1f}"
              f"{('oui' if wt else 'NON'):>9}{('OUI' if coupee else 'non'):>9}"
              f"   {s['note']}")
        lignes.append(dict(s=s, verts=v, faces=f, topo=topo, etanche=wt,
                           coupee=coupee))
    print("-" * 96)

    fermees = [l for l in lignes if l["etanche"] and not l["coupee"]]
    ouvertes = [l for l in lignes if not (l["etanche"] and not l["coupee"])]
    print(f"\nSurfaces FERMÉES (le pipeline s'applique)   : "
          f"{', '.join(l['s']['nom'] for l in fermees)}")
    print(f"Surfaces NON FERMÉES (il ne s'applique pas) : "
          f"{', '.join(l['s']['nom'] for l in ouvertes)}")

    print("""
CE QUE ÇA VEUT DIRE — à lire avant les chiffres de la partie B
--------------------------------------------------------------
Pour la gyroïde et pour la sextique de Barth, le chi affiché n'a AUCUN sens
géométrique : Gauss-Bonnet et chi = 2 - 2g ne valent que sur une surface fermée.
Le maillage étant tronqué par la boîte, le nombre obtenu décrit la découpe.
C'est le test d'étanchéité (colonne « étanche ») qui empêche d'y croire — et
c'est exactement à cela qu'il sert dans src/invariants.py.

  - gyroïde : elle est triplement PÉRIODIQUE. Elle sépare l'espace en deux
    labyrinthes imbriqués et non bornés. Le signe de f distingue bien les deux,
    mais aucun des deux n'est un « intérieur » : il n'y a pas de volume fini.
    Conséquence concrète pour NOTRE pipeline : src/inside_outside.py fabrique
    le signe dedans/dehors à partir d'un nuage orienté, en regardant de quel
    côté de la normale sortante on se trouve — une construction qui n'a de sens
    que si la surface est close. Sur une vraie donnée de gyroïde, l'étiquette
    de signe SERAIT INDISPONIBLE. On l'a ici seulement parce qu'on possède la
    formule. On s'interdit donc de s'en servir : la gyroïde est apprise en
    partie B SANS le terme de signe, avec surface + normales + eikonal
    uniquement, ce qui est le régime pour lequel IGR a été conçu.

  - Barth : deux problèmes cumulés. D'abord 65 points doubles ordinaires, où le
    gradient s'annule — la condition eikonal ‖∇f‖ = 1 y est violée par la VRAIE
    fonction, donc aucune distance signée lisse ne peut représenter cette forme
    exactement. Ensuite, avec w = 1 la surface possède des branches NON BORNÉES
    (asymptotes au cône (φ²x²−y²)(φ²y²−z²)(φ²z²−x²) = 0) : elle n'est contenue
    dans aucune boîte, vérifié de box = 1.0 à box = 3.0. Le signe reste
    défini point par point dans la boîte, donc on l'apprend AVEC le terme de
    signe, mais la topologie de la reconstruction est une propriété de la
    troncature, pas de la surface.

  - Chmutov et metaballs sont les seuls cas fermés — et les seuls où la
    prétention du sujet est réellement testable.
""")
    return lignes


# ==================================================================
# B. Apprentissage
# ==================================================================

def _cle(nom):
    return nom.split()[0].lower().strip("éû")


def apprend(ligne, avec_signe, n_epochs=EPOCHS):
    """
    Entraîne puis mesure une forme — sauf si le résultat est déjà en cache.

    Quatre entraînements de 4000 époques dépassent la durée d'un appel unique.
    Chaque forme est donc figée dans output/cache_forme_*.npz dès qu'elle est
    finie, et relancer le script reprend là où il s'était arrêté.
    """
    nom = ligne["s"]["nom"]
    box = float(ligne["s"]["box"])
    f_exact, n_exact = ligne["s"]["f"], ligne["s"]["n"]

    chemin = f"output/cache_forme_{_cle(nom)}.npz"
    if os.path.exists(chemin):
        d = np.load(chemin, allow_pickle=False)
        r = dict(nom=nom, box=box, avec_signe=bool(d["avec_signe"]),
                 verts=d["verts"], faces=d["faces"],
                 chamfer=float(d["chamfer"]), hausdorff=float(d["hausdorff"]),
                 eikonal=float(d["eikonal"]), chi_appris=int(d["chi_appris"]),
                 chi_exact=int(d["chi_exact"]), fermee=bool(d["fermee"]),
                 n_verts=int(d["n_verts"]), duree=float(d["duree"]))
        print(f"\n--- {nom} : [cache] Chamfer={r['chamfer']:.5f}  "
              f"eikonal={r['eikonal']:.4f}  chi_appris={r['chi_appris']}", flush=True)
        return r

    print(f"\n--- {nom} : apprentissage "
          f"{'AVEC' if avec_signe else 'SANS'} le terme de signe ---", flush=True)
    set_seed(0)
    t0 = time.time()

    # nuage d'ENTRAÎNEMENT et nuage de RÉFÉRENCE : deux tirages indépendants,
    # sinon le Chamfer mesurerait la mémorisation des points fournis.
    P_train = sample_surface(f_exact, n_exact, box, n=15000, seed=0)
    P_ref = sample_surface(f_exact, n_exact, box, n=30000, seed=12345)
    N_train = n_exact(P_train)
    print(f"    nuage entraînement {len(P_train)} pts | référence {len(P_ref)} pts"
          f" | max|f| sur le nuage = {np.abs(f_exact(P_train)).max():.2e}", flush=True)

    X_surf = torch.tensor(P_train, dtype=torch.float32)
    normals = torch.tensor(N_train, dtype=torch.float32)
    rng = np.random.default_rng(1)
    Xv = rng.uniform(-box, box, size=(40000, 3)).astype(np.float32)
    # Seul le SIGNE de f est retenu, jamais sa valeur (règle du projet).
    sv = np.sign(np.asarray(f_exact(Xv))).astype(np.float32)[:, None]
    X_vol = torch.tensor(Xv)
    sign_vol = torch.tensor(sv)

    model = IGRNet(radius_init=0.5 * box)
    train_igr(model, X_surf, normals, X_vol, sign_vol, box,
              n_epochs=n_epochs, batch_surf=4096, batch_vol=4096, lr=1e-3,
              lam_sign=(1.0 if avec_signe else 0.0))

    xs, ys, zs, values = evaluate_grid(model, box, grid_res=RES_APPRIS)
    verts, faces, _, _ = extract_surface(xs, ys, zs, values)
    ch = chamfer_distance(verts, P_ref)
    eik = eikonal_error(model, box, n=50000)
    topo = euler_characteristic(verts, faces)
    dt = time.time() - t0

    print(f"    -> sommets={len(verts)}  Chamfer={ch['chamfer']:.5f}  "
          f"Hausdorff={ch['hausdorff']:.5f}  eikonal={eik:.4f}  "
          f"chi_appris={topo['chi']}  ({dt:.0f}s)", flush=True)

    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), f"output/model_{_cle(nom)}.pt")
    r = dict(nom=nom, box=box, avec_signe=avec_signe, verts=verts,
             faces=faces, chamfer=ch["chamfer"], hausdorff=ch["hausdorff"],
             eikonal=eik, chi_appris=topo["chi"], chi_exact=ligne["topo"]["chi"],
             fermee=(ligne["etanche"] and not ligne["coupee"]),
             n_verts=len(verts), duree=dt)
    np.savez(chemin, **{k: v for k, v in r.items() if k != "nom"})
    return r


def partie_B(lignes):
    print("\n" + "=" * 96)
    print("B. APPRENTISSAGE DES SURFACES COMPLEXES  (IGRNet, %d époques)" % EPOCHS)
    print("=" * 96)

    par_nom = {l["s"]["nom"]: l for l in lignes}
    plan = [("gyroïde", False),            # pas d'intérieur -> pas de signe
            ("Barth sextique", True),      # signe défini dans la boîte
            ("Chmutov degré 4", True),     # fermée, genre 5
            ("metaballs anneau", True)]    # fermée, genre 1

    res = [apprend(par_nom[nom], avec_signe) for nom, avec_signe in plan]

    print("\n" + "-" * 96)
    print(f"{'surface':<22}{'signe':>7}{'fermée':>8}{'Chamfer':>10}"
          f"{'Hausdorff':>11}{'eikonal':>9}{'chi exact':>11}{'chi appris':>12}{'durée':>8}")
    print("-" * 96)
    for r in res:
        chi_ex = str(r["chi_exact"]) if r["fermee"] else f"({r['chi_exact']})"
        print(f"{r['nom']:<22}{('oui' if r['avec_signe'] else 'non'):>7}"
              f"{('oui' if r['fermee'] else 'NON'):>8}"
              f"{r['chamfer']:>10.5f}{r['hausdorff']:>11.5f}{r['eikonal']:>9.4f}"
              f"{chi_ex:>11}{r['chi_appris']:>12d}{r['duree']:>7.0f}s")
    print("-" * 96)
    print("chi entre parenthèses : maillage non fermé, la valeur ne veut rien dire.")
    return res


# ==================================================================
# Figures
# ==================================================================

def figures(lignes, res):
    os.makedirs("output", exist_ok=True)
    produits = []
    for l in lignes:
        m = viz_pv.prepare(viz_pv.to_polydata(l["verts"], l["faces"]))
        p = f"output/forme_{_cle(l['s']['nom'])}.png"
        viz_pv.render(m, p, color="steelblue", show_axes=True,
                      title=f"{l['s']['nom']} — chi mesuré = {l['topo']['chi']}"
                            f"{'' if (l['etanche'] and not l['coupee']) else '  (maillage NON fermé)'}")
        produits.append((p, l["s"]["nom"]))

    for r in res:
        l = next(x for x in lignes if x["s"]["nom"] == r["nom"])
        m_ex = viz_pv.prepare(viz_pv.to_polydata(l["verts"], l["faces"]))
        m_ap = viz_pv.prepare(viz_pv.to_polydata(r["verts"], r["faces"]))
        p = f"output/appris_{_cle(r['nom'])}.png"
        viz_pv.render_comparison(
            [m_ex, m_ap],
            [f"exact — chi = {r['chi_exact']}",
             f"appris — chi = {r['chi_appris']}, Chamfer = {r['chamfer']:.4f}"],
            p, color="steelblue",
            title=f"{r['nom']} : formule contre réseau")
        produits.append((p, f"{r['nom']} exact vs appris"))

    print("\nFigures produites :")
    for p, nom in produits:
        print(f"  {p:<44} {viz_pv.taille_ko(p):8.1f} Ko   {nom}")


def main():
    lignes = partie_A()
    res = partie_B(lignes)
    figures(lignes, res)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
