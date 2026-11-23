# Miscarea robotului pe baza campului potential.
import numpy as np

from fields import forta_atractie, forta_respingere, forta_laterala, RAPORT_RAZA_LATERALA
from geometry import closest_on_polygon


def in_poligon(punct, poligon):
    # True daca punctul e in interiorul poligonului (metoda razei: numaram
    # de cate ori o semidreapta spre dreapta taie laturile poligonului)
    x, y = punct
    n = len(poligon)
    in_interior = False
    for i in range(n):
        x1, y1 = poligon[i]
        x2, y2 = poligon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_taietura = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_taietura:
                in_interior = not in_interior
    return in_interior


def loveste_obstacol(punct, obstacole, toleranta=1e-6):
    # robotul e punctiform, iar obstacolele sunt deja expandate:
    # coliziune = in interiorul unui obstacol sau chiar pe marginea lui
    for poligon in obstacole:
        if in_poligon(punct, poligon):
            return True
        if closest_on_polygon(punct, poligon)[1] < toleranta:
            return True
    return False


# Forta laterala se opreste cand robotul e cu atat mai aproape de tinta decat in punctul unde s-a blocat (a iesit din minim si poate merge din nou spre tinta)
MARGINE_LATERAL = 1.0

# Daca robotul e la mai putin de atat de tinta si totusi se opreste, forta laterala nu il mai
# ajuta (tinta e in raza d0 a unui obstacol): nu o pornim, iar starea e "aproape"
RAZA_TINTA = 1.0


def cost_drum(harta, robot, semn, dist_activare, xi, eta, d0, lat, pas, max_sim):
    # Simuleaza drumul de aici pana la tinta, ocolind spre 'semn'. Costul = lungimea drumului;
    # daca robotul nu ajunge (ramane blocat sau se invarte), costul e infinit.
    ist, st = deplaseaza_robot(harta, xi, eta, d0, lat=lat, pas=pas, max_pasi=max_sim, start=robot,
                               lateral_pornita=True, dist_activare=dist_activare,
                               parte=semn, semn_lat=semn)
    if st not in ("ajuns", "aproape"):
        return np.inf
    return float(np.sum(np.linalg.norm(np.diff(ist["pozitii"], axis=0), axis=1)))


def fara_componenta_spre_obstacole(forta, robot, obstacole, raza):
    # Plasa de siguranta: scoatem din forta laterala orice componenta care ar impinge robotul SPRE
    # un obstacol la care e deja foarte aproape (raza = d0 / 6, cum e linia punctata cea mai
    # apropiata). Ramane doar partea tangenta. Mai departe nu o aplicam: altfel robotul nu mai
    # poate trece prin culoare inguste.
    for poligon in obstacole:
        punct, d = closest_on_polygon(robot, poligon)
        if 0 < d < raza:
            n = (robot - punct) / d                  # directia dinspre obstacol spre robot
            c = np.dot(forta, n)
            if c < 0:
                forta = forta - c * n
    return forta


def alege_semn(harta, robot, obstacole_lat, dist_activare, xi, eta, d0, lat, pas, parte, max_sim=1200):
    # Partea spre care ocoleste forta laterala: +1 = dreapta, -1 = stanga (privind spre obstacol).
    # parte = +1 / -1: impusa din taste.
    # parte = 0: automat. Simulam drumul pana la tinta pe ambele parti si o alegem pe cea mai scurta;
    # daca niciuna nu scapa, alegem partea spre care forta laterala il duce mai aproape de tinta.
    if parte:
        return 1.0 if parte > 0 else -1.0
    tinta = harta["stop"]
    costuri = {s: cost_drum(harta, robot, s, dist_activare, xi, eta, d0, lat, pas, max_sim)
               for s in (1.0, -1.0)}
    if np.isfinite(min(costuri.values())) and costuri[1.0] != costuri[-1.0]:
        return min(costuri, key=costuri.get)
    spre_dreapta = forta_laterala(robot, obstacole_lat, d0, 1.0)
    return 1.0 if np.dot(spre_dreapta, tinta - robot) >= 0 else -1.0


def obstacole_apropiate(robot, obstacole, raza):
    # obstacolele aflate la distanta < raza de robot; cel putin cel mai apropiat
    dist = [closest_on_polygon(robot, poligon)[1] for poligon in obstacole]
    alese = [poligon for poligon, d in zip(obstacole, dist) if d < raza]
    if not alese:
        alese = [obstacole[int(np.argmin(dist))]]
    return alese


def deplaseaza_robot(harta, xi, eta, d0, lat=0.0, pas=0.05, eps_tinta=0.1, max_pasi=3000, start=None,
                     lateral_pornita=False, dist_activare=None, parte=0, semn_lat=None):

    # daca primim o pozitie de start, plecam de acolo; altfel din START-ul hartii
    if start is None:
        start = harta["start"]
    robot = np.array(start, dtype=float)

    tinta = harta["stop"]
    obstacole = harta["obstacles"]
    xmin, xmax, ymin, ymax = harta["bounds"]

    # listele in care salvam istoricul (pentru desen si animatie)
    pozitii = []
    forte_atractie = []
    forte_respingere = []
    forte_laterale = []
    forte_totale = []

    eta_efectiv_lista = []
    lat_activa_lista = []
    dist_activare_lista = []
    raza_lat = RAPORT_RAZA_LATERALA * d0
    lateral_activ = lateral_pornita
    obstacole_lat = []
    semn_activ = 0.0
    semn_lista = []
    if lateral_activ:
        obstacole_lat = obstacole_apropiate(robot, obstacole, raza_lat)
        if dist_activare is None:
            dist_activare = np.linalg.norm(robot - tinta)
        # partea impusa din tasta (P) are prioritate; altfel pastram partea de pana acum
        if parte:
            semn_activ = 1.0 if parte > 0 else -1.0
        else:
            semn_activ = semn_lat if semn_lat else alege_semn(harta, robot, obstacole_lat, dist_activare,
                                                              xi, eta, d0, lat, pas, parte)
    pas_activare = 0    # de la acest pas numaram fereastra de detectare a blocajului

    status = "max_pasi"
    fereastra = 60    # cati pasi ne uitam in urma ca sa detectam blocajul

    for _ in range(max_pasi):
        f_atractie = forta_atractie(robot, tinta, xi)
        distanta = min((closest_on_polygon(robot, poligon)[1] for poligon in obstacole), default=np.inf)
        eta_efectiv = eta
        if distanta < d0 / 6:
            forta_a = np.linalg.norm(f_atractie)
            eta_efectiv = max(eta, distanta + 2 * forta_a, 2 * forta_a / d0)
        f_respingere = forta_respingere(robot, obstacole, d0, eta_efectiv)
        dist_tinta = np.linalg.norm(robot - tinta)
        if lateral_activ and (dist_tinta < dist_activare - MARGINE_LATERAL or dist_tinta < RAZA_TINTA):
            lateral_activ = False
            semn_activ = 0.0
            pas_activare = len(pozitii)

        if lateral_activ:
            f_laterala = forta_laterala(robot, obstacole_lat, d0, semn_activ * lat)    # tangenta la obstacol
            f_laterala = f_laterala * np.linalg.norm(f_atractie) / (RAPORT_RAZA_LATERALA * d0)   # lat = de cate ori e mai mare decat atractia
            f_laterala = fara_componenta_spre_obstacole(f_laterala, robot, obstacole, d0 / 6)
        else:
            f_laterala = np.zeros(2)
        f_total = f_atractie + f_respingere + f_laterala

        # 2. Salvam pozitia si fortele acestui pas
        pozitii.append(robot.copy())
        forte_atractie.append(f_atractie)
        forte_respingere.append(f_respingere)
        forte_laterale.append(f_laterala)
        forte_totale.append(f_total)
        eta_efectiv_lista.append(eta_efectiv)
        lat_activa_lista.append(1.0 if lateral_activ else 0.0)
        semn_lista.append(semn_activ)
        dist_activare_lista.append(dist_activare if lateral_activ else 0.0)

        # 3. Am ajuns la tinta?
        if np.linalg.norm(robot - tinta) < eps_tinta:
            status = "ajuns"
            break

        # 4. Proiectiile rezultantei pe Ox si Oy, si marimea ei
        Fx, Fy = f_total
        marime = np.sqrt(Fx ** 2 + Fy ** 2)

        # 5. Forta zero: atractia si respingerea se anuleaza exact
        if marime < 1e-9:
            if dist_tinta < RAZA_TINTA:
                status = "aproape"
                break
            if not lateral_activ and lat != 0 and len(obstacole):
                lateral_activ = True           # minim exact: pornim forta laterala
                pas_activare = len(pozitii)
                obstacole_lat = obstacole_apropiate(robot, obstacole, raza_lat)
                dist_activare = np.linalg.norm(robot - tinta)
                semn_activ = alege_semn(harta, robot, obstacole_lat, dist_activare, xi, eta, d0, lat, pas, parte)
                continue
            status = "blocat"
            break

        # 6. Un pas de lungime 'pas' pe directia rezultantei, pe fiecare axa:
        #    x <- x + pas * Fx / |F|     y <- y + pas * Fy / |F|
        x_nou = robot[0] + pas * Fx / marime
        y_nou = robot[1] + pas * Fy / marime

        #    robotul ramane in spatiul de lucru: marginile hartii sunt pereti
        x_nou = min(max(x_nou, xmin), xmax)
        y_nou = min(max(y_nou, ymin), ymax)
        robot_nou = np.array([x_nou, y_nou])

        # 7. Coliziune: atractia a invins respingerea si pasul ar intra intr-un obstacol.
        #    Nu mutam robotul: ramane in ultimul punct valid (ultima pozitie salvata).
        #    Mareste eta sau d0, ori micsoreaza xi, ca sa nu mai treaca.
        if loveste_obstacol(robot_nou, obstacole):
            status = "coliziune"
            break

        robot = robot_nou

        # 8. Blocaj: dupa 'fereastra' pasi robotul abia s-a deplasat (minim local)
        if len(pozitii) - pas_activare > fereastra:
            if np.linalg.norm(robot - pozitii[-fereastra]) < 3 * pas:
                if np.linalg.norm(robot - tinta) < RAZA_TINTA:
                    status = "aproape"         # oprit langa tinta: ea e in raza unui obstacol
                    break
                if not lateral_activ and lat != 0 and len(obstacole):
                    lateral_activ = True       # minim local: pornim forta laterala
                    pas_activare = len(pozitii)
                    obstacole_lat = obstacole_apropiate(robot, obstacole, raza_lat)
                    dist_activare = np.linalg.norm(robot - tinta)
                    semn_activ = alege_semn(harta, robot, obstacole_lat, dist_activare, xi, eta, d0, lat, pas, parte)
                else:
                    status = "blocat"          # blocat si cu forta laterala
                    break

    istoric = {
        "pozitii": np.array(pozitii),
        "forta_atractie": np.array(forte_atractie),
        "forta_respingere": np.array(forte_respingere),
        "forta_laterala": np.array(forte_laterale),
        "forta_totala": np.array(forte_totale),
        "eta_efectiv": np.array(eta_efectiv_lista),
        "lat_activa": np.array(lat_activa_lista),
        "dist_activare": np.array(dist_activare_lista),
        "semn_lat": np.array(semn_lista),
    }
    return istoric, status