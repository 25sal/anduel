import csv
import random
import math
import matplotlib.pyplot as plt

# Definizione delle dimensioni del piano (in miglia nautiche)
area_size = 40  # Piano 40 x 40
max_distance = 2.5  # Raggio massimo per il punto di incontro
speed_min = 500  # Velocità minima (km/h)
speed_max = 800  # Velocità massima (km/h)





# Funzione per generare un punto casuale all'interno del piano
def generate_random_point(size):
    return random.uniform(0, size), random.uniform(0, size)

# Funzione per calcolare i punti di ingresso e uscita degli aerei
def calculate_trajectory(p_incontro, area_size, speed,  angle):
    # Calcoliamo la distanza percorsa dall'aereo
    # distance = speed * (time_to_meet / 3600)  # Convertiamo il tempo in ore
    
    # Calcoliamo il punto di ingresso
    y1 = (area_size-p_incontro[0]) *math.tan(angle) + p_incontro[1]
    if y1 > area_size:
        y1 = area_size
        x1 = (area_size-p_incontro[1]) / math.tan(angle) + p_incontro[0]
    else:
        x1 = area_size
        y1 = (area_size-p_incontro[0]) *math.tan(angle) + p_incontro[1]

    ingresso_x = x1
    ingresso_y = y1        
    
   

   
   
    
    # Assicuriamoci che il punto di ingresso sia sul bordo dell'area
    ingresso_x = max(0, min(area_size, ingresso_x))
    ingresso_y = max(0, min(area_size, ingresso_y))
    
    # Calcoliamo il punto di uscita (supponendo che l'aereo prosegua dritto)
    y2 = (0-p_incontro[0]) *math.tan(angle) + p_incontro[1]
    if y2 < 0:
        y2 = 0
        x2 = (0-p_incontro[1]) / math.tan(angle) + p_incontro[0]
    else:
        x2 = 0
        y2 = (0-p_incontro[0]) *math.tan(angle) + p_incontro[1]
    uscita_x = x2
    uscita_y = y2
    
    return (ingresso_x, ingresso_y), (uscita_x, uscita_y)




# Visualizzazione delle traiettorie
def visualizza_traiettorie(aerei_data, area_size, p_incontro):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Disegna il bordo dell'area
    ax.plot([0, area_size, area_size, 0, 0], [0, 0, area_size, area_size, 0], 'k-', lw=2, label="Bordo area")

    # Aggiungi il punto di incontro
    ax.scatter(*p_incontro, color="red", s=100, label="Punto di incontro")
    
    #circle = plt.Circle(p_incontro,max_distance, color='g', fill=True,alpha=0.2)
    #ax.add_patch(circle)

    # Disegna le traiettorie degli aerei
    for i, aereo in enumerate(aerei_data):
        ingresso, uscita, _, speed = aereo
        ax.plot([ingresso[0], uscita[0]], [ingresso[1], uscita[1]], label=f"Aereo {i + 1} (Vel: {speed:.1f} km/h)")
        ax.scatter(*ingresso, color="blue", s=50, label=f"Ingresso Aereo {i + 1}" if i == 0 else None)
        ax.scatter(*uscita, color="green", s=50, label=f"Uscita Aereo {i + 1}" if i == 0 else None)

    # Imposta i limiti dell'area
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect('equal', adjustable='box')

    # Aggiungi titolo, legenda e griglia
    ax.set_title("Traiettorie degli aerei")
    ax.set_xlabel("Miglia nautiche (x)")
    ax.set_ylabel("Miglia nautiche (y)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Mostra il grafico
    plt.show()


# Generazione dei dati
p_incontro = generate_random_point(area_size)  # Punto di incontro casuale


aerei_data = []
for _ in range(2):
    # Scegliamo un angolo casuale per la direzione
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(speed_min, speed_max)  # Velocità casuale
    ingresso, uscita = calculate_trajectory(p_incontro, area_size, speed, angle)
    aerei_data.append((ingresso, uscita, p_incontro, speed))

# Salvataggio in un file CSV
output_file = "traiettorie_aerei.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Ingresso_x", "Ingresso_y", "Uscita_x", "Uscita_y", "PuntoIncontro_x", "PuntoIncontro_y", "Velocità_kmh"])
    for aereo in aerei_data:
        ingresso, uscita, p_incontro, speed = aereo
        writer.writerow([
            ingresso[0], ingresso[1],
            uscita[0], uscita[1],
            p_incontro[0], p_incontro[1],
            speed
        ])



# Visualizza le traiettorie
visualizza_traiettorie(aerei_data, area_size, p_incontro)
