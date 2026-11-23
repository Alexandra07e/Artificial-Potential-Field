#Geometrie: distanta de la un punct la un poligon.
import numpy as np


def closest_on_segment(robot, A, B):
    #Cel mai apropiat punct de pe segmentul AB fata de punctul robot
    AB = B - A
    t = np.clip(np.dot(robot - A, AB) / np.dot(AB, AB), 0.0, 1.0) #aflam pozitia proiectiei robotului pe latura
    #clip - taie valoarea la intervalul [0,1]
    return A + t * AB #incepem din A si facem t*AB - pasi din lungimea laturii


def closest_on_polygon(robot, poligon):
    #Cel mai apropiat punct de pe conturul poligonului.

    cel_mai_apropiat_punct, cea_mai_mica_distanta = None, np.inf
    nr_varfuri = len(poligon)

    #trecerea prin toate punctele din poligon
    for i in range(nr_varfuri):
        # "% nr_varfuri" face ca ultima latura sa se inchida la primul varf
        capat_a = poligon[i]
        capat_b = poligon[(i + 1) % nr_varfuri]

        # Cel mai apropiat punct de pe aceasta latura
        punct = closest_on_segment(robot, capat_a, capat_b)

        # Distanta de la robot la acest punct
        distanta = np.linalg.norm(robot - punct)

        # Daca latura asta e mai aproape decat toate cele de pana acum, o tinem minte
        if distanta < cea_mai_mica_distanta:
            cel_mai_apropiat_punct = punct
            cea_mai_mica_distanta = distanta

    return cel_mai_apropiat_punct, cea_mai_mica_distanta