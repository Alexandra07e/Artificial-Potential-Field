# Potentiale si forte din metoda campului potential.
# Atractia si respingerea sunt proportionale cu distanta (liniare).
import numpy as np

from geometry import closest_on_polygon


# Atractia (spre tinta)
def potential_atractie(robot, tinta, xi):
    # U_a = 1/2 * xi * d_final^2
    distanta_tinta = np.linalg.norm(robot - tinta)
    return 0.5 * xi * distanta_tinta ** 2


def forta_atractie(robot, tinta, xi):
    # F_a = -xi * (C - C_final), vector [Fx, Fy]
    # proportionala cu distanta pana la tinta
    return -xi * (robot - tinta)


# Respingerea (dinspre obstacole)
def potential_respingere(robot, obstacole, d0, eta):
    # U_r = 1/2 * eta * (d0 - d)^2   daca d < d0, altfel 0
    # Potentialele obstacolelor se aduna.
    potential = 0.0
    for poligon in obstacole:
        _, distanta = closest_on_polygon(robot, poligon)
        # obstacolul conteaza doar in raza lui de influenta d0
        if distanta < d0:
            potential += 0.5 * eta * (d0 - distanta) ** 2
    return potential


def forta_respingere(robot, obstacole, d0, eta):
    # F_r = eta * (d0 - d), pe directia dinspre obstacol spre robot

    # Fortele obstacolelor se aduna vectorial. Intoarce [Fx, Fy].
    forta = np.zeros(2)
    for poligon in obstacole:
        punct, distanta = closest_on_polygon(robot, poligon)
        if 0 < distanta < d0:
            # versorul dinspre obstacol spre robot
            directie = (robot - punct) / distanta
            # marimea creste liniar cand robotul se apropie de obstacol
            marime = eta * (d0 - distanta)
            forta += marime * directie
    return forta


# Forta laterala (tangentiala la obstacole; porneste singura cand robotul e blocat)
RAPORT_RAZA_LATERALA = 2.0     # raza ei de actiune = RAPORT_RAZA_LATERALA * d0


def forta_laterala(robot, obstacole, d0, lat):
    raza = RAPORT_RAZA_LATERALA * d0
    forta = np.zeros(2)
    if lat == 0:
        return forta
    for poligon in obstacole:
        a = np.asarray(poligon, dtype=float)
        ab = np.roll(a, -1, axis=0) - a                              # laturile
        t = np.einsum("ij,ij->i", robot - a, ab) / np.einsum("ij,ij->i", ab, ab)
        puncte = a + np.clip(t, 0.0, 1.0)[:, None] * ab               # cel mai apropiat punct de pe fiecare latura
        vect = robot - puncte
        dist = np.linalg.norm(vect, axis=1)
        in_raza = (dist > 0) & (dist < raza)
        if not in_raza.any():
            continue
        alese = in_raza & (t > 0) & (t < 1)                          # laturi cu punctul in interior
        if not alese.any():                                          # doar varfuri: folosim cel mai apropiat
            alese = np.zeros(len(dist), dtype=bool)
            alese[np.argmin(np.where(in_raza, dist, np.inf))] = True
        directie = vect[alese] / dist[alese][:, None]
        pondere = (raza - dist[alese])[:, None]
        forta += (pondere * np.column_stack([-directie[:, 1], directie[:, 0]])).sum(axis=0)
    return lat * forta


# Rezultanta
def forta_totala(robot, obstacole, tinta, xi, eta, d0, lat=0.0):
    # F = F_a + F_r + F_lat
    return (forta_atractie(robot, tinta, xi) + forta_respingere(robot, obstacole, d0, eta)
            + forta_laterala(robot, obstacole, d0, lat))