import numpy as np
from visualize import leggi_dati_csv, visualizza_traiettorie

class AnglePath:
    
    def __init__(self,  conflict_time, stps=1):
        self.len = stps
        self.steps = np.zeros(2*self.len)
        self.deadline = conflict_time 
    
    def random(self):
        for i in range(0,self.len,2):
            # time of the step
            self.steps[i] = np.random.uniform(0, self.deadline)
            # angle of the step
            self.steps[i+1] = {'angle': np.random.uniform(-10, 10)}
        return self.steps
    
    
    
if __name__ == "__main__":
    
    # Lettura dei dati dal file CSV generato
    file_csv = "data/traiettorie_aerei.csv"  # Sostituisci con il percorso corretto
    aerei_data, p_incontro = leggi_dati_csv(file_csv)

    # Dimensione del piano
    area_size = 40
    print(len(aerei_data), len(p_incontro))
    # Visualizza le traiettorie
    visualizza_traiettorie(aerei_data[:2], area_size, p_incontro)
    
    
    iterations = 10
    epsilon = 0.5
    visualizza_traiettorie(aerei_data[:2], area_size, p_incontro)
    for i in range(iterations):
        if np.random(1) < epsilon:
            # compute conflict time
            conflict_time = 10
            seed =  AnglePath(conflict_time=1)
        else:
            #get seed from q-table
            pass
        
        #compute starting population with seed and configuration
        #run genetic (minimize distance from target and, mazimize distance from other vehicles)
        # get pareto front
        # visualize seed and pareto front
        # compute reward
        # update q-table
    