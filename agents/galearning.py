import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import random

# -----------------------------------
# 1. Read CSV Data (Only Two Aircraft)
# -----------------------------------
def leggi_dati_csv(file_path):
    """
    Reads aircraft data from a CSV file and returns:
      - a list of aircraft dictionaries,
      - a list of conflict points (tuples).
    Only the first two aircraft are processed.
    """
    aerei_data = []
    collision_points = []
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ingresso_x = float(row["Ingresso_x"])
            ingresso_y = float(row["Ingresso_y"])
            uscita_x = float(row["Uscita_x"])
            uscita_y = float(row["Uscita_y"])
            p_incontro_x = float(row["PuntoIncontro_x"])
            p_incontro_y = float(row["PuntoIncontro_y"])
            velocita = float(row["Velocità_kmh"])
            
            aereo = {
                "Ingresso_x": ingresso_x,
                "Ingresso_y": ingresso_y,
                "Uscita_x": uscita_x,
                "Uscita_y": uscita_y,
                "Velocità_kmh": velocita
            }
            aerei_data.append(aereo)
            collision_points.append((p_incontro_x, p_incontro_y))
            if len(aerei_data) == 2:
                break
    return aerei_data, collision_points

# -----------------------------------
# 2. Utility Functions
# -----------------------------------
def distance_2d(p1, p2):
    """Euclidean distance in 2D."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def predict_position(aircraft, t_minutes):
    """
    Predicts the aircraft’s (x, y) position at time t_minutes, assuming constant speed 
    along a straight-line from (Ingresso_x, Ingresso_y) to (Uscita_x, Uscita_y).
    """
    x_start = aircraft["Ingresso_x"]
    y_start = aircraft["Ingresso_y"]
    x_end   = aircraft["Uscita_x"]
    y_end   = aircraft["Uscita_y"]
    speed_kmh = aircraft["Velocità_kmh"]
    speed_nm_min = speed_kmh * 0.539957 / 60.0  # km/h -> nm/min

    total_distance = distance_2d((x_start, y_start), (x_end, y_end))
    if speed_nm_min < 1e-6:
        return (x_start, y_start)
    total_time = total_distance / speed_nm_min
    if t_minutes >= total_time:
        return (x_end, y_end)
    fraction = t_minutes / total_time
    x_pos = x_start + fraction * (x_end - x_start)
    y_pos = y_start + fraction * (y_end - y_start)
    return (x_pos, y_pos)

# -----------------------------------
# 3. Conflict Checks
# -----------------------------------
def compute_conflict_time_between(aircraft1, aircraft2, lookahead=8.0, dt=0.1, threshold=7.5):
    """
    Steps through time from 0 to lookahead minutes and checks the distance between aircraft1 and aircraft2.
    If the distance falls below 'threshold' (nm), returns (True, conflict_time, conflict_point)
    where conflict_point is the midpoint at that time.
    """
    t = 0.0
    while t <= lookahead:
        pos1 = predict_position(aircraft1, t)
        pos2 = predict_position(aircraft2, t)
        dist = distance_2d(pos1, pos2)
        # Debug print can be uncommented for detailed output
        # print(f"[DEBUG] t={t:.2f} min, inter-aircraft distance = {dist:.2f} nm")
        if dist < threshold:
            conflict_point = ((pos1[0] + pos2[0]) / 2.0, (pos1[1] + pos2[1]) / 2.0)
            return True, t, conflict_point
        t += dt
    return False, None, None

def check_conflict_to_csv_point(aircraft, conflict_point, threshold=20.0, lookahead=8.0, dt=0.1):
    """
    Steps through time for a given aircraft to check if its distance to 'conflict_point' 
    falls below 'threshold' (nm). Returns (True, conflict_time) if so, else (False, None).
    """
    t = 0.0
    while t <= lookahead:
        pos = predict_position(aircraft, t)
        if distance_2d(pos, conflict_point) < threshold:
            return True, t
        t += dt
    return False, None

# -----------------------------------
# 4. Generate Avoidance Route (with Randomization Option)
# -----------------------------------
def generate_avoidance_route(aircraft, conflict_time, angle_range=(-10,10), duration_range=(1.0,3.0)):
    """
    Generates a modified route for the aircraft that deviates from its original path.
    Randomly selects a deviation angle within angle_range (degrees) and a duration within duration_range (minutes).
    Returns a list of route points: [start, conflict_pos, deviation_pos, final_dest].
    """
    deviation_angle = random.uniform(*angle_range)
    duration = random.uniform(*duration_range)
    print(f"[DEBUG] Random deviation angle: {deviation_angle:.2f} deg, duration: {duration:.2f} min")

    x_start = aircraft["Ingresso_x"]
    y_start = aircraft["Ingresso_y"]
    x_end   = aircraft["Uscita_x"]
    y_end   = aircraft["Uscita_y"]
    speed_kmh = aircraft["Velocità_kmh"]
    speed_nm_min = speed_kmh * 0.539957 / 60.0

    conflict_pos = predict_position(aircraft, conflict_time)

    # Compute original normalized velocity vector
    vx = x_end - x_start
    vy = y_end - y_start
    mag = math.sqrt(vx**2 + vy**2)
    if mag < 1e-6:
        mag = 1.0
    vx /= mag
    vy /= mag

    # Rotate vector by the random deviation_angle (counterclockwise)
    rad = math.radians(deviation_angle)
    vx_rot = vx * math.cos(rad) - vy * math.sin(rad)
    vy_rot = vx * math.sin(rad) + vy * math.cos(rad)

    deviation_distance = speed_nm_min * duration
    deviation_pos = (
        conflict_pos[0] + vx_rot * deviation_distance,
        conflict_pos[1] + vy_rot * deviation_distance
    )

    route = [
        (x_start, y_start),    # start
        conflict_pos,          # start of deviation (aircraft position at conflict_time)
        deviation_pos,         # deviation endpoint (after duration)
        (x_end, y_end)         # final destination
    ]
    print(f"[DEBUG] Generated avoidance route: {route}")
    return route

class AnglePath:
    def __init__(self, conflict_time, stps=1):
        self.len = stps
        self.steps = np.zeros(2 * self.len)
        self.deadline = conflict_time 
    
    def random(self):
        """
        Generate a random deviation path within constraints.
        """
        for i in range(0, self.len * 2, 2):
            self.steps[i] = np.random.uniform(0, self.deadline)
            self.steps[i + 1] = np.random.uniform(-10, 10)
        return self.steps

def plot_all_routes(aerei_data, generated_routes, conflict_point, area_size):
    """Plots original and all generated avoidance routes."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i, aircraft in enumerate(aerei_data):
        start = (aircraft["Ingresso_x"], aircraft["Ingresso_y"])
        end = (aircraft["Uscita_x"], aircraft["Uscita_y"])
        ax.plot([start[0], end[0]], [start[1], end[1]], label=f"Original Route A{i}", linestyle='-', linewidth=2)
        ax.scatter(*start, color="blue", s=50)
        ax.scatter(*end, color="green", s=50)
    
    for i, route in enumerate(generated_routes):
        x_vals, y_vals = zip(*route)
        ax.plot(x_vals, y_vals, linestyle="--", label=f"Avoidance Route {i}")
    
    ax.scatter(conflict_point[0], conflict_point[1], color="red", s=100, label="Conflict Point")
    
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nautical Miles (x)")
    ax.set_ylabel("Nautical Miles (y)")
    ax.set_title("Original and Avoidance Trajectories")
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    file_csv = "data/traiettorie_aerei.csv"
    aerei_data, p_incontro = leggi_dati_csv(file_csv)
    
    area_size = 40
    print("Aircraft Data:", aerei_data)
    print("Conflict Points from CSV:", p_incontro)
    
    dist_initial = distance_2d(
        (aerei_data[0]["Ingresso_x"], aerei_data[0]["Ingresso_y"]),
        (aerei_data[1]["Ingresso_x"], aerei_data[1]["Ingresso_y"])
    )
    print(f"[DEBUG] Initial distance between aircraft: {dist_initial:.2f} nm")
    
    conflict_detected, conflict_time, conflict_point = compute_conflict_time_between(
        aerei_data[0], aerei_data[1], lookahead=60.0, dt=0.1, threshold=7.5
    )
    
    for i, aircraft in enumerate(aerei_data):
        conflict_csv_detected, conflict_csv_time = check_conflict_to_csv_point(
            aircraft, p_incontro[0], threshold=7.5, lookahead=8.0, dt=0.1
        )
        if conflict_csv_detected:
            conflict_detected = True
            conflict_time = conflict_csv_time
            conflict_point = p_incontro[0]
            print(f"[INFO] Conflict detected for Aircraft {i} at CSV conflict point {conflict_point} at t={conflict_time:.2f} min")
    
    generated_routes = []
    if conflict_detected:
        print(f"[INFO] Conflict detected at t={conflict_time:.2f} min at {conflict_point}")
        
        iterations = 10
        epsilon = 0.5
        for i in range(iterations):
            if random.random() < epsilon:
                seed = AnglePath(conflict_time=conflict_time)
                seed_path = seed.random()
                print(f"[DEBUG] Generated seed path {i}: {seed_path}")
                
                dist0 = distance_2d(predict_position(aerei_data[0], conflict_time), conflict_point)
                dist1 = distance_2d(predict_position(aerei_data[1], conflict_time), conflict_point)
                avoiding_aircraft = 0 if dist0 > dist1 else 1
                
                avoidance_route = generate_avoidance_route(
                    aerei_data[avoiding_aircraft], conflict_time, angle_range=(-10,10), duration_range=(1.0,3.0)
                )
                print(f"[INFO] Generated avoidance route for Aircraft {avoiding_aircraft}: {avoidance_route}")
                generated_routes.append(avoidance_route)
			
            else:
                pass
			
			#compute starting population with seed and configuration
			#run genetic (minimize distance from target and, mazimize distance from other vehicles)
			# get pareto front
			# visualize seed and pareto front
			# compute reward
			# update q-table
			
        plot_all_routes(aerei_data, generated_routes, conflict_point, area_size)
        
    else:
        print("[INFO] No conflict detected. No paths generated.")
