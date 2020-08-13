# -*- coding: utf-8 -*-

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# _/  強化学習DQN (Deep Q Network)
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

import numpy as np

from collections import deque
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD

from network import resnet, create_unet

# ４次元ベクトルの任意ch内容確認
def print_state_At(state, index):
    tmp = ''
    for i in range(16):
        for j in range(16):
            if state[0][i][j][index] < 0 : tmp += str('%5.3f' % state[0][i][j][index])+' '
            else                         : tmp += str('%6.3f' % state[0][i][j][index])+' '
        if i < 15 : tmp += '\n'
    print(tmp)


# 次の行動を決める
def action_select(action):
    velocity = 0.5
    if action == 0 : linear = -velocity; angle = -1.0
    if action == 1 : linear = -velocity; angle =  0.0
    if action == 2 : linear = -velocity; angle =  1.0
    if action == 3 : linear =       0.0; angle = -1.0
    if action == 4 : linear =       0.0; angle =  0.0
    if action == 5 : linear =       0.0; angle =  1.0
    if action == 6 : linear =  velocity; angle = -1.0
    if action == 7 : linear =  velocity; angle =  0.0
    if action == 8 : linear =  velocity; angle =  1.0
    return linear, angle

# [1]損失関数の定義
# 損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01):
        self.debug_log = True

        #self.model = create_unet()
        self.model = resnet()

        #self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        #self.optimizer = Adam()

        #self.optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss=huberloss, optimizer=self.optimizer)

        if self.debug_log == True:
            self.model.summary()

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN, bot_color):
        inputs  = np.zeros((batch_size, 16, 16, 7))
        targets = np.zeros((batch_size, 16, 16, 1))
        mini_batch = memory.sample(batch_size)

        #for i, (state_b, linear_b, angle_b, reward_b, next_state_b) in enumerate(mini_batch):
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if reward_b == 0:
            #if not np.sum(next_state_b) == 0: # 状態が全部ゼロじゃない場合
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]    # (16, 16, 1)
                retmainQs = np.reshape(retmainQs, (16, 16))        # (16, 16)

                # 最大の報酬を返す行動を選択する
                next_action = np.unravel_index(np.argmax(retmainQs), retmainQs.shape)
                #if bot_color == 'r' : print_state_At(targetQN.model.predict(next_state_b), 0)

                targetQs    = targetQN.model.predict(next_state_b)[0] # (16, 16, 1)
                targetQs    = np.reshape(targetQs, (16, 16))          # (16, 16, 1)
                next_reward = targetQs[next_action[0]][next_action[1]]

                target = reward_b + gamma * next_reward

            targets[i] = self.model.predict(state_b)               # Qネットワークの出力
            #if bot_color == 'r' : print(i, reward_b, action_b[0], action_b[1], target, targets[i][action_b[0]])

            # 学習する報酬の手調整
            #ban = np.array( [ [4,8], [7,8], [7,7], [8,12], [8,9], [8,8], [8,7], [8,4], [9,9], [9,8], [12,8]  ] )
            for k in range(16):
                for l in range(16):
                    if abs(targets[i][k][l]) > 0.95          : targets[i][k][l] = targets[i][k][l]*0.8  # 大きすぎる場合は少し調整を行う
                    #if k < 1 or l < 1 or k > 14 or l > 14 : targets[i][k][l] = 0                        # 領域外の報酬は０固定
                    #for a in ban:
                    #    if a[0] == k and a[1] == l        : targets[i][k][l] = 0   # 障害物座標の報酬は０固定

            targets[i][action_b[0]][action_b[1]] = target          # 教師信号
            np.set_printoptions(precision=1)
            #if bot_color == 'r' : print(i, reward_b, action_b, target)

        # shiglayさんよりアドバイスいただき、for文の外へ修正しました
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # 初回は時間がかかる epochsは訓練データの反復回数、verbose=0は表示なしの設定
        #self.model.fit(inputs, targets, epochs=1, verbose=1)  # 初回は時間がかかる epochsは訓練データの反復回数、verbose=0は表示なしの設定


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        #self.max_size = max_size
        #self.reset()

    def reset(self):
        pass
        #self.buffer = deque(maxlen=self.max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


# [4]カートの状態に応じて、行動を決定するクラス
# アドバイスいただき、引数にtargetQNを使用していたのをmainQNに修正しました
class Actor:
    def __init__(self):
        self.debug_log = True

    # 移動先をランダムに生成
    def generateRandomDestination(self):

        # 移動禁止箇所
        #ban = np.array( [ [4,8], [7,8], [7,7], [8,12], [8,9], [8,8], [8,7], [8,4], [9,9], [9,8], [12,8]  ] )
        ban = np.array( [ [99,99], [7,7], [8,8] ] )  # [7,7] and [8,8] are reserved.

        flag = True
        while flag:
            flag   = False
            #action = np.array( [int(3+np.random.rand()*11), int(3+np.random.rand()*11)] )
            action = np.array( [1+int(np.random.rand()*14), int(1+np.random.rand()*14)] )
            for a in ban:
                if a[0] == action[0] and a[1] == action[1] : flag = True
        return action

    # ２次元numpy配列でＮ番目に大きい要素を返す
    def getIndexAtMaxN(self, input, N):
        result = np.where(input==np.sort(input.flatten())[-N])
        result = np.array([ result[0][0], result[1][0] ])
        return result

    def get_action(self, state, episode, mainQN, bot_color, action_bf, action_bf2, delta_score, sim_flag, force_random_action, avoid_best_action):   # [C]ｔ＋１での行動を返す

        # 徐々に最適行動のみをとる、ε-greedy法
        #epsilon = 0.001 + 0.9 / (1.0+episode)
        if sim_flag : epsilon = 0.1  # 学習時のランダム係数
        else        : epsilon = 0.00  # 実機ではランダム動作を行わない

        # 移動禁止箇所
        #ban = np.array( [ [4,8], [7,8], [7,7], [8,12], [8,9], [8,8], [8,7], [8,4], [9,9], [9,8], [12,8]  ] )

        predicted = False  # True if predicted by network, otherwise random selection
        if epsilon <= np.random.uniform(0, 1) and not force_random_action:
            #if bot_color == 'r' : print('Learned')
            predicted = True

            retTargetQs = mainQN.model.predict(state)             # (1, 16, 16, 1)
            #if bot_color == 'r' : print_state_At(retTargetQs, 0)  # 予測結果を表示
            #retTargetQs = mainQN.model.predict(state)[0]          # (16, 16, 1)
            retTargetQs = retTargetQs[0]                          # (16, 16, 1)
            retTargetQs = np.reshape(retTargetQs, (16, 16))       # (16, 16)
            action      = np.unravel_index(np.argmax(retTargetQs), retTargetQs.shape)
            action      = np.array(action)

            # 学習結果前フィールドと同じで現状負けていたら２～５番目の候補のどれかに変更する
            same_and_behind = (((action[0] == action_bf[0] and action[1] == action_bf[1]) or (action[0] == action_bf2[0] and action[1] == action_bf2[1])) and delta_score <= 0)

            # Choose action from 2nd to 5th best candidates
            if same_and_behind or avoid_best_action:
                if bot_color == 'r' : print('Select Except Top Action')
                action = self.getIndexAtMaxN(retTargetQs, 2+int(np.random.rand()*9))
                #action = self.generateRandomDestination()

            '''
            # 学習結果が移動禁止箇所だったらランダムを入れておく
            flag   = False
            for a in ban:
                if a[0] == action[0] and a[1] == action[1] : flag = True
            if flag or (action[0] < 3) or (action[1] < 3) or (action[0] > 13) or (action[1] > 13):
                if bot_color == 'r' : print('Random flag=', flag, 'action=', action)
                action = self.generateRandomDestination()
            else:
                if bot_color == 'r' : print('Learned')
            '''

        else:
            if bot_color == 'r' : print('Random')
            # 移動禁止箇所以外へランダムに行動する
            action = self.generateRandomDestination()

        return action, predicted
