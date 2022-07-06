import pygame
from GamePAI_pytorch_dqn_weapon_posses_undivided import GamePAI
import random
import multiprocessing as mp
import pandas as pd
import sys

def Random_Runner(total_episodes,episodes,dfrewards,agents_reward,lock,agent):
    
    steps = 0
    reward = 0
    total_reward = 0
    process_cont = True
    g = 0
    while process_cont:
        limit = 'unreached'
        g += 1
        game = GamePAI(1,'Connan',444,444,1,True,g,False,False,agent)
        done = False
        total_episode_reward = 0
        steps = 0
        while not done:
            steps += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            x = random.randrange(8)
            state, playerstatus, textArray, reward, done= game.playerAction(x)
            total_episode_reward += reward
            if steps > 2000 and game.cave <2:
                print('limit reached')
                limit = 'reached'
                done = True
            if done:
                lock.acquire()
                agents_reward[0] += total_episode_reward
                episodes[0] += 1
                game.gameOver_pytorch()
                total_reward += total_episode_reward
                average_reward = total_reward/(g)
                total_average_reward = agents_reward[0]/episodes[0]
                print("for agent {} episode is {} episode reward is {} total reward for the agent is {} and avg reward is {} total episode number is {} all process reward is {} and total average is {} ".format(agent,g,total_episode_reward, total_reward,average_reward,episodes[0],agents_reward[0],total_average_reward)+limit)
                episodeStat = [agent,g,total_episode_reward, total_reward,average_reward,episodes[0],agents_reward[0],total_average_reward,limit]
                dfrewards.append(episodeStat)
                lock.release()
                if episodes[0] > total_episodes:
                    process_cont = False
                if episodes[0] == total_episodes + mp.cpu_count():
                    d = dfrewards
                    d = list(d)
                    # print(type(d))
                    dfrewards_data_frame = pd.DataFrame(d, columns=['agent','agent episode','episode rewards', 'total agent rewards', 'mean agent rewards','total episodes','total reward','total average reward','limit'])
                    destination = r'D:\ekpa\diplomatiki\date_31_5_22_test_4\agent_random.xlsx'
                    dfrewards_data_frame.to_excel(destination)
                    process_cont = False

                
    print("for agent {} total reward after {} episode is {} and avg reward is {}".format(agent,g, total_reward, average_reward))


if __name__ == '__main__':
    num_processes = mp.cpu_count()
    episodes = mp.Manager().list([0])
    dfrewards = mp.Manager().list()
    agents_reward = mp.Manager().list([0])
    lock = mp.Lock()
    processes = []
    for agent in range(num_processes):
        p = mp.Process(target=Random_Runner, args=(1000,episodes,dfrewards,agents_reward,lock,agent))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


    
