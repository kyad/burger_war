# -*- coding: utf-8 -*-

# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
# _/  強化学習DQN (Deep Q Network)
# _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/

import numpy as np

from collections import deque, namedtuple
import tensorflow as tf
from keras import backend as K
#from keras.optimizers import Adam, SGD

from network import resnet, create_unet
import rospy

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# ４次元ベクトルの任意ch内容確認
def print_state_At(state, index_batch, index):
    tmp = ''
    for i in range(16):
        for j in range(16):
            if state[index_batch][i][j][index] < 0 : tmp += str('%5.3f' % state[index_batch][i][j][index])+' '
            else                         : tmp += str('%6.3f' % state[index_batch][i][j][index])+' '
        if i < 15 : tmp += '\n'
    rospy.loginfo('Batch(%d/%d):' % (index_batch + 1, state.shape[0]))
    #rospy.loginfo(tmp)


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

# def prob_loss(y_true, y_pred):
#     y_t = y_true
#     y_p = y_pred
#     return - K.mean(K.sum(y_t * K.log(y_p), axis=[1, 2, 3]))

# def reward_loss(y_true, y_pred):
#     y_t = y_true
#     y_p = y_pred
#     return K.mean(K.square(y_t - y_p))


# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, debug_log=False, learning_rate=0.01):
        self.debug_log = debug_log

        #self.model = create_unet()
        self.model = resnet(input_shape=(16, 16, 8), num_layers=[3, 4, 3])

        #self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        #self.optimizer = Adam()

        #self.optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.optimizer = tf.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

        #self.model.compile(loss=huberloss, optimizer=self.optimizer)
        #self.model.compile(loss=[prob_loss, reward_loss], optimizer=self.optimizer, loss_weights=[1.0, 1.0])

        if self.debug_log == True:
            self.model.summary()

    # 重みの学習
    def replay(self, memory, batch_size, gamma):
        #inputs  = np.zeros((batch_size, 16, 16, 7))
        #targets = np.zeros((batch_size, 16, 16, 1))

        if len(memory) < batch_size:    # memoryにbatch_size以上のデータが保存されているか確認
            print("memory size = {} is smaller than batch size = {}.".format(len(memory), batch_size))
            batch_size = len(memory)

        mini_batch = memory.sample(batch_size)

        # それぞれのデータを結合する
        state_batch = np.concatenate(mini_batch.state)                          # (batch_size, 16, 16, 8)
        action_batch = np.concatenate(mini_batch.action).reshape(batch_size, 2)       # (batch_size, 2)
        reward_batch = np.array(mini_batch.reward).reshape(batch_size, 1)       # (batch_size, 1)
        next_state_batch = np.concatenate(mini_batch.next_state)  # (batch_size, 16, 16, 8)

        # 教師データの作成
        pred = self.model.predict(next_state_batch).max(1).max(1)                   # (batch_size, 1)
        next_state_values = pred.reshape(pred.shape[0])                             # (batch_size,)
        y_target = reward_batch.reshape(batch_size) + gamma * next_state_values     # (batch_size,)
        y_target = np.clip(y_target, -1.0, 1.0)
        y_target = tf.convert_to_tensor(y_target, dtype=tf.float32)                 # (batch_size,)
        
        # actionのindexを作成
        zero_idx = np.zeros((batch_size, 1)).astype('int32')                        # (batch_size, 1)
        arange_idx = np.arange(batch_size).reshape(batch_size, 1).astype('int32')   # (batch_size, 1)
        action_batch_idx = np.concatenate([arange_idx, action_batch, zero_idx], 1)  # (batch_size, 1) + (batch_size, 2) + (batch_size, 1) -> (batch_size, 4)
        action_batch_idx = tf.convert_to_tensor(action_batch_idx, dtype=tf.int32)   # (batch_size, 4), (batch_id, action_x, action_y, 0)がbatch_size分

        # バッチ学習
        #loss = self.model.train_on_batch(x=state_batch, y=[prob_true, reward_true])

        # GradientTapeでy_predとlossを定義し、学習を実行する
        with tf.GradientTape() as tape:
            y_pred = self.model(state_batch.astype(np.float32)) # (batch_size, 16, 16, 1)
            y_pred = tf.gather_nd(y_pred, action_batch_idx)     # (batch_size,)
            loss = huberloss(y_target, y_pred)

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

        # #for i, (state_b, linear_b, angle_b, reward_b, next_state_b) in enumerate(mini_batch):
        # for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
        #     inputs[i:i + 1] = state_b
        #     target = reward_b

        #     if reward_b == 0:
        #     #if not np.sum(next_state_b) == 0: # 状態が全部ゼロじゃない場合
        #         # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
        #         retmainQs = self.model.predict(next_state_b)[0]    # (16, 16, 1)
        #         retmainQs = np.reshape(retmainQs, (16, 16))        # (16, 16)

        #         # 最大の報酬を返す行動を選択する
        #         next_action = np.unravel_index(np.argmax(retmainQs), retmainQs.shape)
        #         #if bot_color == 'r' : print_state_At(targetQN.model.predict(next_state_b), 0)

        #         targetQs    = targetQN.model.predict(next_state_b)[0] # (16, 16, 1)
        #         targetQs    = np.reshape(targetQs, (16, 16))          # (16, 16, 1)
        #         next_reward = targetQs[next_action[0]][next_action[1]]

        #         target = reward_b + gamma * next_reward

        #     targets[i] = self.model.predict(state_b)               # Qネットワークの出力
        #     #if bot_color == 'r' : print(i, reward_b, action_b[0], action_b[1], target, targets[i][action_b[0]])

        #     # 学習する報酬の手調整
        #     #ban = np.array( [ [4,8], [7,8], [7,7], [8,12], [8,9], [8,8], [8,7], [8,4], [9,9], [9,8], [12,8]  ] )
        #     for k in range(16):
        #         for l in range(16):
        #             if abs(targets[i][k][l]) > 0.95          : targets[i][k][l] = targets[i][k][l]*0.8  # 大きすぎる場合は少し調整を行う
        #             #if k < 1 or l < 1 or k > 14 or l > 14 : targets[i][k][l] = 0                        # 領域外の報酬は０固定
        #             #for a in ban:
        #             #    if a[0] == k and a[1] == l        : targets[i][k][l] = 0   # 障害物座標の報酬は０固定

        #     targets[i][action_b[0]][action_b[1]] = target          # 教師信号
        #     np.set_printoptions(precision=1)
        #     #if bot_color == 'r' : print(i, reward_b, action_b, target)

        # # shiglayさんよりアドバイスいただき、for文の外へ修正しました
        # self.model.fit(inputs, targets, batch_size=batch_size, epochs=1, verbose=1)  # 初回は時間がかかる epochsは訓練データの反復回数、verbose=0は表示なしの設定


# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        '''
        buffer: deque(state, action, next_state, reward)
            - state: 4D array<float>, (1, 16, 16, 7)
            - action: 1D array<int> (x,y), (2,)
            - reward: int 
            - next_state: 4D array<float>, (1, 16, 16, 7)
        '''
        self.buffer = deque(maxlen=max_size)
        #self.max_size = max_size
        #self.reset()

    def reset(self):
        # 前の試合のmemoryも学習に使うため、memoryは消さない
        pass
        #self.buffer = deque(maxlen=self.max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        samples = [self.buffer[ii] for ii in idx]
        return Transition(*zip(*samples))

    def __len__(self):
        return len(self.buffer)


# [4]カートの状態に応じて、行動を決定するクラス
# アドバイスいただき、引数にtargetQNを使用していたのをmainQNに修正しました
class Actor:
    def __init__(self, debug_log=False):
        self.debug_log = debug_log

    # 移動先をランダムに生成
    def generateRandomDestination(self):

        # 移動禁止箇所
        #ban = np.array( [ [4,8], [7,8], [7,7], [8,12], [8,9], [8,8], [8,7], [8,4], [9,9], [9,8], [12,8]  ] )
        ban = np.array( [ [99,99], [7,7], [8,8] ] )  # [7,7] and [8,8] are reserved.

        flag = True
        while flag:
            flag   = False
            #action = np.array( [1+int(np.random.rand()*14), int(1+np.random.rand()*14)] )
            action = np.array( [int(np.random.rand()*16), int(np.random.rand()*16)] )
            for a in ban:
                if a[0] == action[0] and a[1] == action[1] : flag = True
        return action

    # ２次元numpy配列でＮ番目に大きい要素を返す
    def getIndexAtMaxN(self, input, N):
        result = np.where(input==np.sort(input.flatten())[-N])
        result = np.array([ result[0][0], result[1][0] ])
        return result

    def get_action(self, state, episode, mainQN, bot_color, action_bf, action_bf2, delta_score, training, force_random_action, avoid_best_action):   # [C]ｔ＋１での行動を返す

        # 徐々に最適行動のみをとる、ε-greedy法
        #epsilon = 0.001 + 0.9 / (1.0+episode)
        if training : epsilon = 0.1  # 学習時のランダム係数
        else        : epsilon = 0.00  # 実機ではランダム動作を行わない

        # 移動禁止箇所
        #ban = np.array( [ [4,8], [7,8], [7,7], [8,12], [8,9], [8,8], [8,7], [8,4], [9,9], [9,8], [12,8]  ] )

        retTargetQs = mainQN.model.predict(state)             # (1, 16, 16, 1), (1, 1)
        #if bot_color == 'r' : print_state_At(retTargetQs, 0, 0)  # 予測結果を表示
        #retTargetQs = mainQN.model.predict(state)[0]          # (16, 16, 1)
        retTargetQs = retTargetQs[0]                          # (16, 16, 1)
        retTargetQs = np.reshape(retTargetQs, (16, 16))       # (16, 16)

        predicted = False  # True if predicted by network, otherwise random selection
        if epsilon <= np.random.uniform(0, 1) and not force_random_action:
            #if bot_color == 'r' : print('Learned')
            predicted = True

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
            #action = self.generateRandomDestination()
            action = self.getIndexAtMaxN(retTargetQs, 2+int(np.random.rand()*9))

        return action, predicted
