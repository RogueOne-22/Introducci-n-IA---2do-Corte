import math
import random
import matplotlib.pyplot as plt

ciudades = {
    'A': (0, 0),
    'B': (1, 5),
    'C': (2, 3),
    'D': (5, 2),
    'E': (6, 6),
    'F': (7, 1),
    'G': (8, 4),
    'H': (9, 9)
}

# ------ Funciones ------
def distancia(a, b):
    x1, y1 = ciudades[a]
    x2, y2 = ciudades[b]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def distancia_ruta(ruta):
    total = 0
    for i in range(len(ruta)):
        total += distancia(ruta[i], ruta[(i + 1) % len(ruta)])  # Regresa al inicio
    return total

def crear_poblacion_inicial(tam_poblacion):
    rutas = []
    lista_ciudades = list(ciudades.keys())
    for _ in range(tam_poblacion):
        ruta = lista_ciudades[:]
        random.shuffle(ruta)
        rutas.append(ruta)
    return rutas

def seleccion(poblacion, distancias):
    fitness = [1/d for d in distancias]  # menor distancia = mayor fitness
    total = sum(fitness)
    probs = [f/total for f in fitness]
    elegido = random.choices(poblacion, weights=probs, k=1)[0]
    return elegido


def cruce(padre1, padre2):
    inicio, fin = sorted(random.sample(range(len(padre1)), 2))
    hijo = [None] * len(padre1)
    hijo[inicio:fin] = padre1[inicio:fin]

    pos = fin
    for ciudad in padre2:
        if ciudad not in hijo:
            if pos >= len(padre2):
                pos = 0
            hijo[pos] = ciudad
            pos += 1
    return hijo


def mutacion(ruta, tasa_mutacion=0.1):
    for i in range(len(ruta)):
        if random.random() < tasa_mutacion:
            j = random.randint(0, len(ruta)-1)
            ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta


def algoritmo_genetico(tam_poblacion=50, generaciones=200, tasa_mutacion=0.1):
    poblacion = crear_poblacion_inicial(tam_poblacion)

    for gen in range(generaciones):
        distancias = [distancia_ruta(r) for r in poblacion]
        nueva_poblacion = []

        for _ in range(tam_poblacion):
            padre1 = seleccion(poblacion, distancias)
            padre2 = seleccion(poblacion, distancias)
            hijo = cruce(padre1, padre2)
            hijo = mutacion(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

   #Mejor ruta
    distancias = [distancia_ruta(r) for r in poblacion]
    mejor_idx = distancias.index(min(distancias))
    return poblacion[mejor_idx], min(distancias)


def graficar_ruta(ruta):
    x = [ciudades[c][0] for c in ruta] + [ciudades[ruta[0]][0]]
    y = [ciudades[c][1] for c in ruta] + [ciudades[ruta[0]][1]]

    plt.figure(figsize=(6,6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    for ciudad, (cx, cy) in ciudades.items():
        plt.text(cx+0.2, cy+0.2, ciudad, fontsize=12, color="red")

    plt.title("Mejor ruta encontrada (TSP)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    mejor_ruta, mejor_distancia = algoritmo_genetico()
    print("Mejor ruta encontrada:", mejor_ruta)
    print("Distancia total:", mejor_distancia)
    graficar_ruta(mejor_ruta)
