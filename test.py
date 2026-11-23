from maps import get_map
from planner import deplaseaza_robot

for nume in ["3", "5", "U"]:
    harta = get_map(nume)
    istoric, status = deplaseaza_robot(harta, xi=1.0, eta=10.0, d0=1.0)
    print(nume, status, len(istoric["pozitii"]), istoric["pozitii"][-1])