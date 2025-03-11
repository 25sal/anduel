import numpy as np
from visualize import leggi_dati_csv, visualizza_traiettorie

class AnglePath:
    
    def __init__(self,  conflict_time, stps=1):
        self.len = stps
        #self.steps = np.zeros(2*self.len)
        self.steps = np.empty(2 * self.len, dtype=object)
        self.deadline = conflict_time 
    
    def random(self):
        for i in range(0,2*self.len,2):
            # time of the step
            self.steps[i] = np.random.uniform(0, self.deadline)
            # angle of the step
            self.steps[i+1] = {'angle': np.random.uniform(-10, 10)}
        return self.steps
    
def generate_2_segment_trajectory(
    conflict_time, 
    stps=2, 
    speed=1.0, 
    dt=0.1, 
    initial_position=(0, 0)
):
    """
    Generates a broken trajectory with exactly 2 segments.
    It uses AnglePath to get two time points and two angle values,
    then simulates the trajectory (x, y) from t=0 to t=conflict_time.
    Returns a dictionary with keys 'x', 'y', and 't'.
    """
    # Create a seed with two time-angle pairs.
    path = AnglePath(conflict_time=conflict_time, stps=stps)
    steps = path.random()  # e.g., [time1, {'angle': angle1}, time2, {'angle': angle2}]
    
    # Extract times and angles from the seed.
    time1, angle1_dict, time2, angle2_dict = steps
    angle1 = angle1_dict['angle']
    angle2 = angle2_dict['angle']
    
    # Sort the two times so we know which one applies first.
    t_breaks = sorted([time1, time2])
    
    # Initialize position and lists for x, y, and t.
    x, y = initial_position
    x_list, y_list = [x], [y]
    t_list = [0.0]
    
    # Simulate the trajectory from t=0 to t=conflict_time in small increments.
    times = np.arange(0, conflict_time + dt, dt)
    for t in times[1:]:
        # Decide which angle applies based on the current time.
        if t < t_breaks[0]:
            angle = angle1
        elif t < t_breaks[1]:
            angle = angle2
        else:
            angle = angle2  # Remain on the second segment
        
        # Convert angle (in degrees) to radians.
        rad = np.radians(angle)
        # Update position.
        x += speed * np.cos(rad) * dt
        y += speed * np.sin(rad) * dt
        
        # Store the results.
        x_list.append(x)
        y_list.append(y)
        t_list.append(t)
    
    return {'x': x_list, 'y': y_list, 't': t_list}    
    
if __name__ == "__main__":
    
    # Lettura dei dati dal file CSV generato
    file_csv = "data/traiettorie_aerei.csv"  # Sostituisci con il percorso corretto
    aerei_data, p_incontro = leggi_dati_csv(file_csv)

    # Dimensione del piano
    area_size = 40
    print(len(aerei_data), len(p_incontro))
    # Visualizza le traiettorie
    visualizza_traiettorie(aerei_data[:2], area_size, p_incontro)
    
    conflict_time = 10
    iterations = 10
    epsilon = 0.5
    
    # Use the speed values from the CSV file for simulation.
    csv_speed1 = aerei_data[0][3]
    csv_speed2 = aerei_data[1][3]
    
    a1 = generate_2_segment_trajectory(
        conflict_time=conflict_time,
        stps=2,
        speed=csv_speed1,     
        dt=0.1,
        initial_position=(0, 0)
    )
    a2 = generate_2_segment_trajectory(
        conflict_time=conflict_time,
        stps=2,
        speed=csv_speed2,      
        dt=0.1,
        initial_position=(10, 10)
    )
    
    # Visualize the newly generated broken trajectories.
    broken_trajectories = [a1, a2]
    visualizza_traiettorie(broken_trajectories, area_size, p_incontro)
    
    for i in range(iterations):
        #if np.random(1) < epsilon:
        if np.random.rand() < epsilon:
            # compute conflict time
            #conflict_time = 10
            #seed =  AnglePath(conflict_time=1)
            seed = AnglePath(conflict_time=conflict_time, stps=2)
            # Optionally, use the seed to generate a new trajectory.
            a1 = generate_2_segment_trajectory(conflict_time, stps=2, speed=csv_speed1, dt=0.1, initial_position=(0, 0))
            a2 = generate_2_segment_trajectory(conflict_time, stps=2, speed=csv_speed2, dt=0.1, initial_position=(10, 10))
            new_trajectories = [a1, a2]
            visualizza_traiettorie(new_trajectories, area_size, p_incontro)
        else:
            #get seed from q-table
            pass
        
        #compute starting population with seed and configuration
        #run genetic (minimize distance from target and, mazimize distance from other vehicles)
        # get pareto front
        # visualize seed and pareto front
        # compute reward
        # update q-table
    