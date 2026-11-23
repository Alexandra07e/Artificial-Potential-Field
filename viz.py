# Desenul static: harta, obstacole, liniile de camp si traiectoria robotului.
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.path import Path

from maps import get_map
from geometry import closest_on_polygon
from parameters import PARAMETRI, modifica, schimba_parte, NUME_PARTE
from planner import deplaseaza_robot, in_poligon

# Cum apare starea robotului in titlu
STATUS_TEXT = {"ajuns": "a ajuns", "blocat": "e blocat (minim local)",
               "aproape": "s-a oprit langa tinta (tinta e in raza d0 a unui obstacol: micsoreaza d0)",
               "coliziune": "a lovit un obstacol", "max_pasi": "a depasit numarul maxim de pasi"}

# Cat de aproape de un obstacol poate fi START / STOP (robotul e punctiform)
MARGINE_PUNCT = 0.15

# Viteza animatiei
PASI_PE_CADRU = 1     # cati pasi ai robotului sare animatia la fiecare cadru
INTERVAL_MS = 150      # pauza intre cadre, in milisecunde (mai mare = mai lent)


def grila_distante(harta, rezolutie=80):
    # Pentru fiecare punct al unei grile calculam distanta pana la cel mai apropiat obstacol.
    # Liniile de camp repulsiv sunt curbele pe care aceasta distanta e constanta.
    # Calculul e vectorizat (toate punctele grilei si toate laturile deodata), ca sa mearga
    # repede si dupa fiecare obstacol adaugat din mouse.
    xmin, xmax, ymin, ymax = harta["bounds"]
    xs = np.linspace(xmin, xmax, rezolutie)
    ys = np.linspace(ymin, ymax, rezolutie)
    X, Y = np.meshgrid(xs, ys)
    puncte = np.column_stack([X.ravel(), Y.ravel()])

    dmin = np.full(len(puncte), 1e3)          # fara obstacole: distanta foarte mare
    in_obstacol = np.zeros(len(puncte), dtype=bool)
    for poligon in harta["obstacles"]:
        a = np.asarray(poligon, dtype=float)
        ab = np.roll(a, -1, axis=0) - a                                   # laturile poligonului
        ap = puncte[:, None, :] - a[None, :, :]                           # (puncte, laturi, 2)
        t = np.einsum("nmk,mk->nm", ap, ab) / np.einsum("mk,mk->m", ab, ab)[None, :]
        proiectie = a[None, :, :] + np.clip(t, 0.0, 1.0)[:, :, None] * ab[None, :, :]
        d = np.linalg.norm(puncte[:, None, :] - proiectie, axis=2).min(axis=1)
        dmin = np.minimum(dmin, d)
        # in interiorul obstacolelor nu desenam linii de camp: mascam acele puncte
        in_obstacol |= Path(np.vstack([a, a[0]])).contains_points(puncte)

    distante = np.ma.masked_where(in_obstacol.reshape(X.shape), dmin.reshape(X.shape))
    return X, Y, distante


def titlu_baza(harta, parametri):
    return "Harta %s  (xi=%.1f, eta=%.1f, d0=%.1f, lat=%.1f, partea: %s)" % (
        harta["name"], parametri["xi"], parametri["eta"], parametri["d0"], parametri["lat"],
        NUME_PARTE[parametri["parte"]])


def deseneaza(ax, harta, parametri, grila, istoric, status, traiectorie_completa=True):
    X, Y, distante = grila
    d0 = parametri["d0"]
    start, tinta = harta["start"], harta["stop"]
    xmin, xmax, ymin, ymax = harta["bounds"]

    # 1. Linii de camp repulsiv (rosu): distante egale fata de obstacole
    if len(harta["obstacles"]):
        ax.contour(X, Y, distante, levels=[d0 / 3, 2 * d0 / 3, d0],
                   colors="#e8503a", linestyles="--", linewidths=1.3)

    # 2. Linii de camp atractiv (verde): cercuri in jurul tintei
    for raza in (2, 4, 6, 8, 10):
        ax.add_patch(Circle(tinta, raza, fill=False, ls="--", lw=1.1, ec="#2e7d32"))

    # 3. Obstacolele
    for poligon in harta["obstacles"]:
        ax.add_patch(Polygon(poligon, closed=True, fc="#a6cdf7", ec="#4a7fb5", lw=1.5, zorder=3))

    # 4. Traiectoria robotului (in animatie o desenam pe masura ce robotul inainteaza)
    if traiectorie_completa:
        pozitii = istoric["pozitii"]
        ax.plot(pozitii[:, 0], pozitii[:, 1], "k-", lw=3, zorder=4)

    # 5. START si STOP
    ax.plot(*start, "o", ms=14, mfc="#4f9a6a", mec="black", zorder=5)
    ax.text(start[0], start[1] + 0.4, "START", ha="center", fontweight="bold")
    ax.plot(*tinta, "h", ms=16, mfc="#f0a030", mec="black", zorder=5)
    ax.text(tinta[0], tinta[1] + 0.4, "STOP", ha="center", fontweight="bold")

    # peretii spatiului de lucru + putina marja, ca robotul lipit de perete sa se vada intreg
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec="black", lw=3, zorder=6))
    marja = 0.4
    ax.set_xlim(xmin - marja, xmax + marja)
    ax.set_ylim(ymin - marja, ymax + marja)
    ax.set_aspect("equal")
    ax.set_title(titlu_baza(harta, parametri))


def anima(fig, ax, istoric, status, titlu, de_la=0, scara=None):
    # Robotul se misca pe traiectorie, iar sagetile fortelor se recalculeaza la fiecare pas.
    # de_la = pasul de la care porneste animatia (0 = de la START).
    pozitii = istoric["pozitii"]
    f_atractie = istoric["forta_atractie"]
    f_respingere = istoric["forta_respingere"]
    f_laterala = istoric["forta_laterala"]
    lat_activa = istoric["lat_activa"]
    semn_lat = istoric["semn_lat"]
    f_total = istoric["forta_totala"]

    # Scara sagetilor: cea mai mare forta din drum are 1.5 unitati pe harta.
    # Aceeasi scara pentru toate trei, ca sa se vada raportul dintre ele.
    # Daca primim scara de la o animatie anterioara, o pastram: asa, cand schimbi xi sau eta,
    # sagetile devin vizibil mai lungi sau mai scurte.
    if scara is None:
        forta_max = max(np.linalg.norm(f, axis=1).max() for f in (f_atractie, f_respingere, f_laterala, f_total))
        scara = 1.5 / forta_max

    # Obiectele care se schimba de la un cadru la altul
    urma, = ax.plot([], [], "k-", lw=3, zorder=4)
    robot, = ax.plot([], [], "o", ms=12, mfc="#1e6fe0", mec="black", zorder=10)
    culori = {"atractie": "#2e7d32", "respingere": "#e8503a", "laterala": "#e08a00", "rezultanta": "#1e6fe0"}
    sageti = {}
    for nume, culoare in culori.items():
        sageti[nume] = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=9,
                                   arrowprops=dict(arrowstyle="-|>", color=culoare, lw=2.8,
                                                   shrinkA=0, shrinkB=0, mutation_scale=18))
    text = ax.text(0.98, 0.02, "", transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                   bbox=dict(fc="white", ec="0.7", alpha=0.9), zorder=11)

    legenda = ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label="Forta " + n) for n, c in culori.items()],
                        loc="upper left", fontsize=9)
    legenda.set_zorder(20)       # legenda ramane deasupra peretilor

    # Pastram pasul la care se afla robotul, ca sa putem continua de acolo
    progres = {"i": de_la, "scara": scara}

    def actualizeaza(i):
        # i = al catelea pas al robotului desenam acum
        progres["i"] = i
        poz = pozitii[i]

        urma.set_data(pozitii[:i + 1, 0], pozitii[:i + 1, 1])
        robot.set_data([poz[0]], [poz[1]])

        for nume, forta in (("atractie", f_atractie[i]), ("respingere", f_respingere[i]),
                            ("laterala", f_laterala[i]), ("rezultanta", f_total[i])):
            sageti[nume].xy = poz + scara * forta            # varful sagetii
            sageti[nume].set_position(poz)                   # baza sagetii (pe robot)
            sageti[nume].set_visible(np.linalg.norm(forta) > 1e-6)

        text.set_text("pas %d\n|F atractie|   = %.2f\n|F respingere| = %.2f\n|F laterala|   = %.2f  (%s)\n|F rezultanta| = %.2f" % (
            i, np.linalg.norm(f_atractie[i]), np.linalg.norm(f_respingere[i]),
            np.linalg.norm(f_laterala[i]), ("activa, spre " + ("dreapta" if semn_lat[i] > 0 else "stanga")) if lat_activa[i] else "inactiva",
            np.linalg.norm(f_total[i])))

        # Mesajul din titlu descrie ce se intampla ACUM, nu cum se termina drumul:
        # starea finala apare doar cand robotul a ajuns la ultimul pas.
        if i < len(pozitii) - 1:
            mesaj = "robotul se misca"
            if lat_activa[i]:
                mesaj += " - forta laterala activa (minim local), ocoleste spre " + ("dreapta" if semn_lat[i] > 0 else "stanga")
        else:
            mesaj = "robotul " + STATUS_TEXT.get(status, status)
            if lat_activa[i] and status == "blocat":
                mesaj += " chiar si cu forta laterala"
        ax.set_title(titlu + "\n" + mesaj)

    # Cadrele sunt chiar pasii care urmeaza, de la 'de_la' pana la ultimul
    cadre = list(range(de_la, len(pozitii), PASI_PE_CADRU))
    if cadre[-1] != len(pozitii) - 1:
        cadre.append(len(pozitii) - 1)
    animatie = FuncAnimation(fig, actualizeaza, frames=cadre, interval=INTERVAL_MS, repeat=False)
    return animatie, progres


# Tastele care schimba parametrii: tasta -> (parametrul, sens)
ACTIUNI = {
    "up": ("xi", +1), "down": ("xi", -1),       # forta de atractie
    "right": ("d0", +1), "left": ("d0", -1),    # distanta de influenta a respingerii
    "w": ("eta", +1), "s": ("eta", -1),         # forta de respingere
    "d": ("lat", +1), "a": ("lat", -1),         # intensitatea fortei laterale: mai mica / mai mare
}
AJUTOR = [
    "\u2191\u2193 atractie (xi)   \u2190\u2192 distanta de influenta (d0)   W/S respingere (eta)   A/D intensitate forta laterala   P partea ocolirii",
    "Shift = pas mare   R = de la start   Spatiu = pauza   3 / 5 / U = harta   C = harta initiala   Z = anuleaza ultimul obstacol",
    "Mouse:  O = deseneaza obstacole (click = varf, Enter / click dreapta = inchide, Esc = gata)   click dreapta pe obstacol = sterge   I = alege START (reporneste)   F = alege STOP",
]


def construieste(nume_harta="3"):
    # Dezactivam scurtaturile matplotlib care s-ar amesteca cu tastele noastre
    # (P = pan, O = zoom, F = ecran complet etc.) sau cu click-urile pe harta
    for scurtatura in ("keymap.back", "keymap.forward", "keymap.save", "keymap.home", "keymap.pan",
                       "keymap.zoom", "keymap.fullscreen", "keymap.grid", "keymap.grid_minor",
                       "keymap.xscale", "keymap.yscale", "keymap.help"):
        if scurtatura in plt.rcParams:
            plt.rcParams[scurtatura] = []

    fig, ax = plt.subplots(figsize=(8, 8.5))
    fig.subplots_adjust(bottom=0.17)
    for k, linie in enumerate(reversed(AJUTOR)):
        fig.text(0.5, 0.01 + 0.025 * k, linie, ha="center", fontsize=7.5)
    mesaj_text = fig.text(0.5, 0.088, "", ha="center", fontsize=9, color="#b03a2e", fontweight="bold")

    # Tot ce se schimba cat timp programul ruleaza. Obstacolele, START si STOP sunt aici
    # (nu doar in maps.py), ca sa le putem schimba din mouse.
    stare = {"harta": nume_harta, "parametri": dict(PARAMETRI), "grila": None,
             "animatie": None, "progres": None, "istoric": None, "scara": None, "pauza": False,
             "obstacole": [], "start": None, "stop": None, "limite": None,
             "mod": None, "puncte": [], "prev": None, "pauza_mod": False, "murdar": False,
             "repornire": False, "stop_nou": False}

    def spune(text):
        mesaj_text.set_text(text)
        fig.canvas.draw_idle()

    def incarca_harta(nume):
        # Punctul de plecare: harta din maps.py (apoi o putem modifica din mouse)
        h = get_map(nume)
        stare["harta"] = nume
        stare["obstacole"] = [np.array(p, dtype=float) for p in h["obstacles"]]
        stare["start"] = np.array(h["start"], dtype=float)
        stare["stop"] = np.array(h["stop"], dtype=float)
        stare["limite"] = h["bounds"]
        stare["grila"] = None
        stare["scara"] = None

    def harta_curenta():
        # harta din maps.py, dar cu obstacolele / START / STOP curente
        h = dict(get_map(stare["harta"]))
        h["obstacles"] = list(stare["obstacole"])
        h["start"] = stare["start"]
        h["stop"] = stare["stop"]
        return h

    def opreste_animatia():
        # La final animatia se opreste singura si isi sterge cronometrul (event_source = None),
        # asa ca verificam inainte sa-l oprim.
        anim = stare["animatie"]
        if anim is not None and anim.event_source is not None:
            anim.event_source.stop()

    def reincarca(de_la_capat, reseteaza_lateral=False):
        # Recalculeaza traiectoria cu parametrii curenti.
        # de_la_capat=False: robotul continua din locul in care se afla acum.
        # de_la_capat=True:  robotul porneste din nou de la START.
        # reseteaza_lateral=True: forta laterala se reevalueaza (de exemplu dupa ce s-a mutat STOP).
        harta = harta_curenta()
        if stare["grila"] is None:                  # distantele depind doar de obstacole
            stare["grila"] = grila_distante(harta)

        vechi = stare["istoric"]
        if de_la_capat or vechi is None:
            i = 0
            istoric, status = deplaseaza_robot(harta, **stare["parametri"])
        else:
            i = stare["progres"]["i"]                         # pasul la care e robotul acum
            continua_lat = bool(vechi["lat_activa"][i]) and not reseteaza_lateral
            nou, status = deplaseaza_robot(harta, start=vechi["pozitii"][i],
                                           lateral_pornita=continua_lat,
                                           dist_activare=(vechi["dist_activare"][i] or None) if continua_lat else None,
                                           semn_lat=(vechi["semn_lat"][i] or None) if continua_lat else None,
                                           **stare["parametri"])
            # pastram trecutul (pasii de dinainte de i) si lipim viitorul recalculat
            istoric = {cheie: np.concatenate([vechi[cheie][:i], nou[cheie]]) for cheie in nou}
        stare["istoric"] = istoric

        opreste_animatia()                          # oprim animatia veche
        ax.clear()
        stare["prev"] = None                        # previzualizarea obstacolului s-a sters odata cu harta
        deseneaza(ax, harta, stare["parametri"], stare["grila"], istoric, status,
                  traiectorie_completa=False)
        stare["animatie"], stare["progres"] = anima(fig, ax, istoric, status, titlu_baza(harta, stare["parametri"]),
                                                  de_la=i, scara=stare["scara"])
        stare["scara"] = stare["progres"]["scara"]
        stare["pauza"] = False
        fig.canvas.draw_idle()

    # ---------- modurile de desen (obstacol / START / STOP) ----------
    def sterge_previzualizare():
        if stare["prev"] is not None:
            try:
                stare["prev"].remove()
            except (ValueError, NotImplementedError):
                pass
            stare["prev"] = None
        fig.canvas.draw_idle()

    def actualizeaza_previzualizare():
        pct = np.array(stare["puncte"], dtype=float).reshape(-1, 2)
        if stare["prev"] is None:
            stare["prev"], = ax.plot([], [], "o-", color="#7a3db8", lw=2, ms=6, zorder=12)
        stare["prev"].set_data(pct[:, 0], pct[:, 1])
        fig.canvas.draw_idle()

    def intra_in_mod(mod, text):
        anim = stare["animatie"]
        # robotul sta pe loc cat desenam; il repornim la iesire
        stare["pauza_mod"] = anim is not None and anim.event_source is not None and not stare["pauza"]
        opreste_animatia()
        stare["mod"], stare["puncte"], stare["murdar"] = mod, [], False
        stare["repornire"], stare["stop_nou"] = False, False
        spune(text)

    def iese_din_mod():
        stare["mod"], stare["puncte"] = None, []
        sterge_previzualizare()
        if stare["murdar"]:
            # Robotul NU o ia de la capat: continua din locul in care se afla, cu harta noua.
            # Doar daca s-a mutat START reporneste (START e chiar punctul de plecare).
            stare["murdar"] = False
            stare["grila"] = None
            repornire, stop_nou = stare["repornire"], stare["stop_nou"]
            stare["repornire"], stare["stop_nou"] = False, False
            reincarca(de_la_capat=repornire, reseteaza_lateral=stop_nou)
            spune("Robotul reporneste de la noul START" if repornire else "Robotul continua din locul in care se afla")
        else:
            anim = stare["animatie"]
            if stare["pauza_mod"] and anim is not None and anim.event_source is not None:
                anim.event_source.start()
            spune("")

    def punct_liber(p):
        # START / STOP nu au voie in obstacol sau lipite de el
        for poligon in stare["obstacole"]:
            if in_poligon(p, poligon) or closest_on_polygon(p, poligon)[1] < MARGINE_PUNCT:
                return False
        return True

    def pozitie_robot():
        # unde se afla robotul chiar acum (cadrul curent al animatiei)
        return stare["istoric"]["pozitii"][stare["progres"]["i"]]

    def inchide_obstacol():
        pct = stare["puncte"]
        if len(pct) < 3:
            spune("Un obstacol are nevoie de cel putin 3 varfuri")
            return
        poligon = np.array(pct, dtype=float)
        x, y = poligon[:, 0], poligon[:, 1]
        aria = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        stare["puncte"] = []
        sterge_previzualizare()
        if aria < 0.05:
            spune("Obstacol respins: e prea mic")
            return
        for nume in ("start", "stop"):
            if in_poligon(stare[nume], poligon) or closest_on_polygon(stare[nume], poligon)[1] < MARGINE_PUNCT:
                spune("Obstacol respins: acopera %s (muta-l intai cu tasta %s)" % (
                    nume.upper(), "I" if nume == "start" else "F"))
                return
        robot_acum = pozitie_robot()
        if in_poligon(robot_acum, poligon) or closest_on_polygon(robot_acum, poligon)[1] < MARGINE_PUNCT:
            spune("Obstacol respins: acopera robotul (pune-l mai departe de el)")
            return
        stare["obstacole"].append(poligon)
        stare["murdar"] = True
        ax.add_patch(Polygon(poligon, closed=True, fc="#a6cdf7", ec="#4a7fb5", lw=1.5, zorder=3))
        spune("Obstacol adaugat (%d in total). Mai deseneaza sau apasa Esc: robotul continua de unde e" % len(stare["obstacole"]))

    def la_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        bara = getattr(fig.canvas, "toolbar", None)
        if bara is not None and getattr(bara, "mode", ""):        # zoom / pan activ: nu e desen
            return
        xmin, xmax, ymin, ymax = stare["limite"]
        p = np.array([min(max(event.xdata, xmin), xmax), min(max(event.ydata, ymin), ymax)])
        mod = stare["mod"]

        if mod == "obstacol":
            if event.button == 3:                               # click dreapta: inchide poligonul
                inchide_obstacol()
            elif event.button == 1:
                pct = stare["puncte"]
                if len(pct) >= 3 and np.linalg.norm(p - pct[0]) < 0.3:   # click langa primul varf: inchide
                    inchide_obstacol()
                else:
                    pct.append(p)
                    actualizeaza_previzualizare()
        elif mod in ("start", "stop"):
            if event.button != 1:
                return
            alt = stare["stop"] if mod == "start" else stare["start"]
            if not punct_liber(p):
                spune("Punct respins: e in obstacol sau prea aproape de el")
            elif np.linalg.norm(p - alt) < 0.5:
                spune("Punct respins: START si STOP sunt prea apropiate")
            else:
                stare[mod] = p
                stare["murdar"] = True
                if mod == "start":
                    stare["repornire"] = True       # START nou = punct de plecare nou
                else:
                    stare["stop_nou"] = True        # STOP nou: robotul continua spre el
                iese_din_mod()
        elif event.button == 3:                                 # in mod normal: click dreapta pe obstacol = sterge
            for k in range(len(stare["obstacole"]) - 1, -1, -1):
                if in_poligon(p, stare["obstacole"][k]):
                    del stare["obstacole"][k]
                    stare["grila"] = None
                    reincarca(de_la_capat=False)
                    spune("Obstacol sters (au mai ramas %d); robotul continua de unde e" % len(stare["obstacole"]))
                    break

    def la_tasta(event):
        if event.key is None:
            return
        tasta = event.key
        # Shift: pentru litere matplotlib da litera mare, pentru sageti "shift+up" etc.
        mare = tasta.startswith("shift+") or (len(tasta) == 1 and tasta.isupper())
        tasta = tasta.replace("shift+", "").lower()

        if stare["mod"] is not None:                 # cat timp desenam, tastele nu schimba parametrii
            if tasta == "escape":
                if stare["mod"] == "obstacol" and stare["puncte"]:
                    stare["puncte"] = []             # prima apasare anuleaza poligonul inceput
                    sterge_previzualizare()
                    spune("Obstacol anulat")
                else:
                    iese_din_mod()
            elif tasta in ("enter", "return") and stare["mod"] == "obstacol":
                if stare["puncte"]:
                    inchide_obstacol()
                else:
                    iese_din_mod()
            elif tasta == "o" and stare["mod"] == "obstacol":
                iese_din_mod()
            elif tasta == "backspace" and stare["puncte"]:
                stare["puncte"].pop()                # sterge ultimul varf
                actualizeaza_previzualizare()
            else:
                spune("Esti in modul de desen: termina-l cu Esc")
            return

        if tasta in ACTIUNI:
            nume, semn = ACTIUNI[tasta]
            modifica(stare["parametri"], nume, semn, 5 if mare else 1)
            reincarca(de_la_capat=False)             # robotul continua de unde e
        elif tasta == "p":                           # partea ocolirii: auto / stanga / dreapta
            schimba_parte(stare["parametri"])
            reincarca(de_la_capat=False)
        elif tasta == "r":                           # valorile initiale, de la START
            stare["parametri"] = dict(PARAMETRI)
            stare["scara"] = None
            reincarca(de_la_capat=True)
        elif tasta in ("3", "5", "u"):               # alta harta, de la START
            incarca_harta(tasta.upper())
            reincarca(de_la_capat=True)
            spune("")
        elif tasta == "c":                           # harta initiala din maps.py (fara ce am desenat)
            incarca_harta(stare["harta"])
            reincarca(de_la_capat=True)
            spune("Harta initiala")
        elif tasta == "z":                           # anuleaza ultimul obstacol
            if stare["obstacole"]:
                stare["obstacole"].pop()
                stare["grila"] = None
                reincarca(de_la_capat=False)
                spune("Ultimul obstacol anulat (au mai ramas %d); robotul continua de unde e" % len(stare["obstacole"]))
        elif tasta == "o":
            intra_in_mod("obstacol", "OBSTACOL: click = varf nou, Enter / click dreapta = inchide, Esc = gata")
        elif tasta == "i":
            intra_in_mod("start", "Click pe harta pentru noul START (Esc = renunt)")
        elif tasta == "f":
            intra_in_mod("stop", "Click pe harta pentru noul STOP (Esc = renunt)")
        elif tasta == " ":                           # pauza / continuare
            anim = stare["animatie"]
            if anim.event_source is not None:         # daca animatia s-a terminat, nu e nimic de oprit
                stare["pauza"] = not stare["pauza"]
                if stare["pauza"]:
                    anim.event_source.stop()
                else:
                    anim.event_source.start()

    fig.canvas.mpl_connect("key_press_event", la_tasta)
    fig.canvas.mpl_connect("button_press_event", la_click)
    stare["la_click"] = la_click
    incarca_harta(nume_harta)
    reincarca(de_la_capat=True)
    return fig, la_tasta, stare


def main():
    fig, la_tasta, stare = construieste("U")     # pastram variabilele, altfel animatia se opreste
    plt.show()


if __name__ == "__main__":
    main()