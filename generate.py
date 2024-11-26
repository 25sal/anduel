import csv
import random
import math

# Definizione delle dimensioni del piano (in miglia nautiche)
area_size = 40  # Piano 40 x 40
max_distance = 2.5  # Raggio massimo per l'incontro
speed_min = 500  # Velocità minima (km/h)
speed_max = 800  # Velocità massima (km/h)
n_experiments = 10


# Funzione per generare un punto casuale all'interno del piano
def generate_random_point(size):
    return random.uniform(0, size), random.uniform(0, size)

# Funzione per calcolare l'intersezione con il bordo dell'area
def calculate_border_intersection(x, y, angle, size):
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    # Troviamo i punti di intersezione con i bordi
    t_x_min = -x / dx if dx != 0 else float('inf')
    t_x_max = (size - x) / dx if dx != 0 else float('inf')
    t_y_min = -y / dy if dy != 0 else float('inf')
    t_y_max = (size - y) / dy if dy != 0 else float('inf')
    
    # Calcoliamo i tempi di uscita più piccoli e positivi
    t_enter = max(min(t_x_min, t_x_max), min(t_y_min, t_y_max))
    t_exit = min(max(t_x_min, t_x_max), max(t_y_min, t_y_max))
    
    # Punti di ingresso e uscita
    x_in, y_in = x + t_enter * dx, y + t_enter * dy
    x_out, y_out = x + t_exit * dx, y + t_exit * dy
    
    return (x_in, y_in), (x_out, y_out)


aerei_data = []

for _ in range(n_experiments):
    # Generazione dei dati per due aerei
    p_incontro = generate_random_point(area_size)
    for _ in range(2):
        angle = random.uniform(0, 2 * math.pi)  # Direzione casuale
        speed = random.uniform(speed_min, speed_max)  # Velocità casuale
        ingresso, uscita = calculate_border_intersection(p_incontro[0], p_incontro[1], angle, area_size)
        aerei_data.append((ingresso, uscita, p_incontro, speed))

# Salvataggio in un file CSV
output_file = "data/traiettorie_aerei.csv"
id  = 0
experiment = 0
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Ingresso_x", "Ingresso_y", "Uscita_x", "Uscita_y", "PuntoIncontro_x", "PuntoIncontro_y", "Velocità_kmh"])
    
    for aereo in aerei_data:
        ingresso, uscita, p_incontro, speed = aereo
        writer.writerow([
            ingresso[0], ingresso[1],
            uscita[0], uscita[1],
            p_incontro[0], p_incontro[1],
            speed, str(experiment)+"_"+str(id)
        ])
        id += 1%2
    experiment += 1

print(f"File CSV generato: {output_file}")
