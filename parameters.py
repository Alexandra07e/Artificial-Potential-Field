# Parametrii pe care ii modificam din tastatura.

# valorile cu care porneste programul
PARAMETRI = {
    "xi": 1.0,     # coeficientul fortei de atractie
    "eta": 10.0,   # coeficientul fortei de respingere
    "d0": 1.0,     # distanta de influenta a respingerii
    "lat": 4.0,    # forta laterala = de cate ori e mai mare decat atractia (lat > 2 invinge atractia); porneste SINGURA in minim local; 0 = oprita
    "parte": 0,    # partea ocolirii: 0 = automat (spre tinta), -1 = stanga, +1 = dreapta (tasta P)
    "pas": 0.05,   # lungimea unui pas al robotului
}

# intre ce valori are voie sa se miste fiecare parametru
LIMITE = {
    "xi": (0.1, 5.0),
    "eta": (1.0, 50.0),
    "d0": (0.2, 3.0),
    "lat": (0.0, 10.0),
}

# cu cat se schimba parametrul la o apasare de tasta
INCREMENT = {
    "xi": 0.1,
    "eta": 1.0,
    "d0": 0.1,
    "lat": 0.5,
}


def modifica(parametri, nume, semn, multiplicator=1):
    # Mareste (semn = +1) sau micsoreaza (semn = -1) un parametru.
    # multiplicator > 1 inseamna pas mai mare (de exemplu cu Shift).
    minim, maxim = LIMITE[nume]
    nou = parametri[nume] + semn * multiplicator * INCREMENT[nume]
    parametri[nume] = round(min(max(nou, minim), maxim), 2)   # ramane in limite


# partea ocolirii se schimba ciclic: automat -> stanga -> dreapta -> automat
PARTI = [0, -1, 1]
NUME_PARTE = {0: "auto", -1: "stanga", 1: "dreapta"}


def schimba_parte(parametri):
    i = PARTI.index(parametri["parte"])
    parametri["parte"] = PARTI[(i + 1) % len(PARTI)]