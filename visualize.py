import csv
import matplotlib.pyplot as plt

# Funzione per leggere i dati dal file CSV
def leggi_dati_csv(file_path):
    """
    Legge i dati degli aerei da un file CSV e li restituisce come lista.

    :param file_path: Percorso al file CSV.
    :return: Lista con i dati degli aerei e il punto di incontro.
    """
    aerei_data = []
    collision_points = []
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ingresso = (float(row["Ingresso_x"]), float(row["Ingresso_y"]))
            uscita = (float(row["Uscita_x"]), float(row["Uscita_y"]))
            p_incontro = (float(row["PuntoIncontro_x"]), float(row["PuntoIncontro_y"]))
            velocita = float(row["Velocità_kmh"])
            aerei_data.append((ingresso, uscita, p_incontro, velocita))
            collision_points.append(p_incontro)
    return aerei_data, collision_points

# Funzione per visualizzare le traiettorie
def visualizza_traiettorie(aerei_data, area_size, p_incontro):
    """
    Visualizza le traiettorie degli aerei su un piano con Matplotlib.

    :param aerei_data: Lista contenente i dati degli aerei (ingresso, uscita, velocità).
    :param area_size: Dimensione del piano (lato del quadrato).
    :param p_incontro: Coordinate del punto di incontro.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Disegna il bordo dell'area
    ax.plot([0, area_size, area_size, 0, 0], [0, 0, area_size, area_size, 0], 'k-', lw=2, label="Bordo area")

    for i in range(0,len(aerei_data),2):
        # Aggiungi il punto di incontro
        ax.scatter(*p_incontro[i], color="red", s=100, label="Punto di incontro")

        # Disegna le traiettorie degli aerei
        for j, aereo in enumerate(aerei_data[i:i+2]):
            ingresso, uscita, _, speed = aereo
            distance = ((ingresso[0] - p_incontro[i][0])**2 + (ingresso[1] - p_incontro[i][1])**2)**0.5
            time_flight = distance / speed
            ax.plot([ingresso[0], uscita[0]], [ingresso[1], uscita[1]], label=f"Aereo {int(i/2)}_{j} (Vel: {speed:.1f} km/h) t={time_flight:.2f}h")
            ax.scatter(*ingresso, color="blue", s=50, label=f"Ingresso Aereo {int(i/2)}_{j}" if i == 0 else None)
            ax.scatter(*uscita, color="green", s=50, label=f"Uscita Aereo {int(i/2)}_{j}" if i == 0 else None)

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

# Lettura dei dati dal file CSV generato
file_csv = "data/traiettorie_aerei.csv"  # Sostituisci con il percorso corretto
aerei_data, p_incontro = leggi_dati_csv(file_csv)

# Dimensione del piano
area_size = 40
print(len(aerei_data), len(p_incontro))
# Visualizza le traiettorie
visualizza_traiettorie(aerei_data[:2], area_size, p_incontro)
