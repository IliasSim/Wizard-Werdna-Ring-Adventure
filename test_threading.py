truncate = 32
n_step = 0
h_step = 4
training_list = []
total_steps = 0
epsilon = 1
gamma = 0.99
for i in range(20000):
    total_steps += 1
    if total_steps%1100 == 0 and total_steps !=0:
            epsilon = epsilon*gamma
            print(epsilon)