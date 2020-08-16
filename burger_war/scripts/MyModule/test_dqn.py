import numpy as np

import DQN

NUM_EPISODE = 10
NUM_STEP = 100

if __name__ == "__main__":
    
    mainQN = DQN.QNetwork(debug_log=True)
    memory = DQN.Memory(max_size=1000)
    #actor = DQN.Actor()

    for episode in range(NUM_EPISODE):
        
        print('episode {}'.format(episode))
        
        state = np.random.rand(16*16*7).reshape(1, 16, 16, 7)
        #action1 = [7, 7]
        #action2 = [8, 8]

        for step in range(NUM_STEP):

            #action, _ = actor.get_action(state, step, mainQN, 'r', action1, action2, 1, True, False, False)
            action = np.array([0,0])

            if step == NUM_STEP - 1:
                next_state = np.zeros((1,16,16,7))
                reward = 1.0
            else:
                next_state = np.random.rand(16*16*7).reshape(1, 16, 16, 7)
                reward = 0.0

            memory.add((state, action, reward, next_state))

            state = next_state 

        print('start learning')
        for epoch in range(10):
            loss = mainQN.replay(memory, 40, 0.97)
            print('epoch:{}, loss:{}'.format(epoch, loss))