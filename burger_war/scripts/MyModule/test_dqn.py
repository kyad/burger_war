import numpy as np
from tqdm import tqdm

import DQN

NUM_EPISODE = 10
NUM_STEP = 100

if __name__ == "__main__":
    
    mainQN = DQN.QNetwork()
    targetQN = DQN.QNetwork()
    memory = DQN.Memory(max_size=1000)
    actor = DQN.Actor()

    targetQN.model.set_weights(mainQN.model.get_weights())

    for episode in range(NUM_EPISODE):
        
        print('episode {}'.format(episode))
        
        state = np.random.rand(16*16*7).reshape(1, 16, 16, 7)

        for step in range(NUM_STEP):

            action, _ = actor.get_action(state, mainQN, 'r', True, False)

            if step == NUM_STEP - 1:
                next_state = np.zeros((1,16,16,7))
                reward = 1.0
            else:
                next_state = np.random.rand(16*16*7).reshape(1, 16, 16, 7)
                reward = 0.0

            memory.add((state, action, next_state, reward))

            state = next_state 

        print('start learning')
        for epoch in tqdm(range(10)):
            mainQN.replay(memory, 40, 0.97, targetQN)

        targetQN.model.set_weights(mainQN.model.get_weights())