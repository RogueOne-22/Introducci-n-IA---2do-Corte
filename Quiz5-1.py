import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------
# Parámetros del problema
# -----------------------------
AREA_SIZE = 5000          # 5 km = 5000 m
N_DRONES = 10             # Número de drones
SENSOR_RADIUS = 200       # Rango de detección (m)
MAX_ITER = 60             # 60 iteraciones = 120 minutos (2 min por iteración)

# Crear un "mapa de probabilidad" para las zonas con supervivientes
def generar_mapa_probabilidad(grid_size=100):
    x = np.linspace(0, AREA_SIZE, grid_size)
    y = np.linspace(0, AREA_SIZE, grid_size)
    X, Y = np.meshgrid(x, y)

    # Definir "zonas calientes" con más probabilidad
    zonas = [
        (1500, 1500, 800),
        (3500, 1200, 600),
        (2500, 4000, 700)
    ]

    prob = np.zeros_like(X)
    for (cx, cy, sigma) in zonas:
        prob += np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    return X, Y, prob / np.max(prob)

# Calcular el "score" de una posición según el mapa
def evaluar_posicion(pos, X, Y, prob):
    dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
    mask = dist <= SENSOR_RADIUS
    return np.sum(prob[mask])

# -----------------------------
# PSO para coordinar drones
# -----------------------------
class DroneSearchPSO:
    def __init__(self, n_drones, max_iter):
        self.n_drones = n_drones
        self.max_iter = max_iter
        self.X, self.Y, self.prob_map = generar_mapa_probabilidad()

        # Inicialización aleatoria
        self.positions = np.random.uniform(0, AREA_SIZE, (n_drones, 2))
        self.velocities = np.random.uniform(-50, 50, (n_drones, 2))
        self.personal_best_pos = np.copy(self.positions)
        self.personal_best_scores = np.array(
            [evaluar_posicion(p, self.X, self.Y, self.prob_map) for p in self.positions]
        )

        self.global_best_pos = self.personal_best_pos[np.argmax(self.personal_best_scores)]
        self.global_best_score = np.max(self.personal_best_scores)

        # Guardar historial
        self.history = []

    def search(self):
        w, c1, c2 = 0.7, 1.5, 1.5  # parámetros PSO

        for t in range(self.max_iter):
            r1, r2 = np.random.rand(), np.random.rand()

            # Actualizar velocidades y posiciones
            self.velocities = (
                w * self.velocities
                + c1 * r1 * (self.personal_best_pos - self.positions)
                + c2 * r2 * (self.global_best_pos - self.positions)
            )
            self.positions += self.velocities

            # Mantener dentro del área
            self.positions = np.clip(self.positions, 0, AREA_SIZE)

            # Evaluar
            scores = [evaluar_posicion(p, self.X, self.Y, self.prob_map) for p in self.positions]

            for i, score in enumerate(scores):
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_pos[i] = self.positions[i]

            # Actualizar mejor global
            if np.max(scores) > self.global_best_score:
                self.global_best_score = np.max(scores)
                self.global_best_pos = self.positions[np.argmax(scores)]

            # Guardar estado
            self.history.append(np.copy(self.positions))

        return self.global_best_pos, self.global_best_score

    def animate_search(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.contourf(self.X, self.Y, self.prob_map, levels=50, cmap="YlOrRd")
        scat = ax.scatter([], [], c="blue", s=40, label="Drones")
        ax.scatter(*self.global_best_pos, c="green", s=100, marker="*", label="Mejor posición")
        ax.set_xlim(0, AREA_SIZE)
        ax.set_ylim(0, AREA_SIZE)
        ax.set_title("Búsqueda de supervivientes")
        ax.legend()

        def update(frame):
            scat.set_offsets(self.history[frame])
            minutos = frame * 2  # cada iteración = 2 minutos
            ax.set_title(f"Búsqueda de supervivientes - Tiempo: {minutos} min")
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=300, blit=True)
        plt.show()
        return ani

# -----------------------------
# Ejecutar simulación
# -----------------------------
if __name__ == "__main__":
    search = DroneSearchPSO(N_DRONES, MAX_ITER)
    best_pos, best_score = search.search()

    print(f"\n🚁 Mejor posición encontrada: {best_pos}")
    print(f"🔍 Score asociado: {best_score:.3f}")

    # Animación
    search.animate_search()
