#!/usr/bin/env python
# -*- coding: utf-8 -*-

# rosservice call /gazebo/set_logger_level ros.move_base WARN   # [INFO] Log Disappear

import os
import tf
import csv
import json
import rospy
import random
import subprocess
import numpy as np
import sys
import datetime
import threading
import time

from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Vector3, Quaternion, PoseWithCovarianceStamped
from sensor_msgs.msg import Image
import actionlib # RESPECT @seigot
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal # RESPECT @seigot
from nav_msgs.msg import Path
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rosparam
 
# 強化学習DQN (Deep Q Network)
from MyModule import DQN

import tensorflow
graph = tensorflow.get_default_graph()

timeScale  = 1    # １秒間で何回座標計算するか？
#timeScale  = 4    # １秒間で何回座標計算するか？ 
#fieldScale = 1.5  # 競技場の広さ
fieldScale = 2.4  # 競技場の広さ
#turnEnd    = 10   # 何ターンで１試合を終了させるか
TimeLimit = 180
#TimeLimit = 30
maxGoalItrCount = 10  # Max number of iteration to determine goal. (STAY will be chosen if exceeded)
maxSameGoalCount = 3  # Max number to approve same goal with previous one.
marginFromObstacle = 0.6  # Margin to be kept from obstacle surface (unit: cell. Field width 2.4 meters is equal to 16 cells)

# クォータニオンからオイラー角への変換
def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0]*180/np.pi, y=e[1]*180/np.pi, z=e[2]*180/np.pi)


# 座標回転行列を返す
def get_rotation_matrix(rad):
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return rot


# 現在地を２次元ベクトル(n*n)にして返す
def get_pos_matrix(x, y, n=16):
    #my_pos  = np.array([self.pos[0], self.pos[1]])           # 現在地点
    pos     = np.array([x, y])                                # 現在地点
    rot     = get_rotation_matrix(-45 * np.pi / 180)          # 45度回転行列の定義
    rotated = np.dot(rot, pos) / fieldScale + 0.5         # 45度回転して最大幅1.5で正規化(0-1)
    #rotated = ( np.dot(rot, pos) + 1 ) / 2                    # 回転を行って0-1の範囲にシフト
    pos_np  = np.zeros([n, n])
    i = int(rotated[0]*n)
    j = int(rotated[1]*n)
    if i < 0: i = 0
    if i > 15: i = 15
    if j < 0: j = 0
    if j > 15: j = 15
    pos_np[i][j] = 1
    #print('*****', x, y, i, j)
    return pos_np


# 移動先の座標
def get_destination(action, n=16):
    # Create table for remapping the specified unreachable action to new reachable one.
    # Note that only one side is defined (x < y) as world is symmetric.
    dest_table = [
        # [ [from_x_int, from_y_int], [to_x_flt, to_y_flt] ], ...

        # CornerBox1
        [ [2,7], [2.0 - marginFromObstacle, 7.0 - marginFromObstacle] ],
        [ [2,8], [2.0 - marginFromObstacle, 8.0 + marginFromObstacle] ],
        [ [3,7], [3.0 + marginFromObstacle, 7.0 - marginFromObstacle] ],
        [ [3,8], [3.0 + marginFromObstacle, 8.0 + marginFromObstacle] ],

        # CornerBox2
        [ [7,12], [7.0 - marginFromObstacle, 12.0 - marginFromObstacle] ],
        [ [7,13], [7.0 - marginFromObstacle, 13.0 + marginFromObstacle] ],
        [ [8,12], [8.0 + marginFromObstacle, 12.0 - marginFromObstacle] ],
        [ [8,13], [8.0 + marginFromObstacle, 13.0 + marginFromObstacle] ],

        # Surface of CenterBox
        [ [6,7], [6.0 - marginFromObstacle, 7.0 - marginFromObstacle] ],
        [ [6,8], [6.0 - marginFromObstacle, 8.0 + marginFromObstacle] ],
        [ [7,9], [7.0 - marginFromObstacle, 9.0 + marginFromObstacle] ],
        [ [8,9], [8.0 + marginFromObstacle, 9.0 + marginFromObstacle] ],

        # Inside CenterBox. Need extra margin.
        [ [7,7], [7.0 - marginFromObstacle - 0.5, 7.0 - marginFromObstacle - 0.5] ],  # This cell is dedicated for loss function and not available. But implemented just in case.
        [ [7,8], [7.0 - marginFromObstacle - 0.5, 8.0 + marginFromObstacle + 0.5] ],
        [ [8,8], [8.0 - marginFromObstacle + 0.5, 8.0 - marginFromObstacle + 0.5] ],  # This cell is dedicated for loss function and not available. But implemented just in case.
    ]
    # Wall1
    dest_table.extend([
        [[0, i], [marginFromObstacle, float(i)]] for i in range(16)
        ])
    # Wall2
    dest_table.extend([
        [[i, 15], [float(i), 15.0 - marginFromObstacle]] for i in range(16)
        ])

    # Remap unreachable action
    action_f = np.array(action, dtype='float32')
    for row in dest_table:
        for i in range(2):
            if row[0][i] == action[0] and row[0][1 - i] == action[1]:
                action_f[0] = row[1][i]
                action_f[1] = row[1][1 - i]
                rospy.logwarn('Unreachable action is remapped: (%d, %d) -> (%.2f, %.2f)' % (action[0], action[1], action_f[0], action_f[1]))
            if row[0][0] == row[0][1]:
                break

    pos   = action_f/n + 1.0/(2*n)
    pos   = (pos-1.0/2)*fieldScale
    rot   = get_rotation_matrix(45 * np.pi / 180)             # 45度回転行列の定義
    return np.dot(rot, pos)                                   # 45度回転


# 自分が向いている向きを２次元ベクトル(n*n)にして返す
def get_ang_matrix(angle, n=16):
    while angle > 0 : angle -= 360
    while angle < 0 : angle += 360
    my_ang  = np.zeros([n, n])
    for i in range(16):
        for j in range(16):
            if 360-22.5 < angle or angle <= 22.5 :              #   0°
                if 10 <= i and 10 <= j      : my_ang[i][j] = 1
            if  45-22.5 < angle <=  45+22.5 :                   #  45°
                if 10 <= i and  5 <= j <= 10: my_ang[i][j] = 1
            if  90-22.5 < angle <=  90+22.5 :                   #  90°
                if 10 <= i and  5 >= j      : my_ang[i][j] = 1
            if 135-22.5 < angle <= 135+22.5 :                   # 135°
                if  5 <= i <=10 and  5 >= j : my_ang[i][j] = 1
            if 180-22.5 < angle <= 180+22.5 :                   # 180°
                if  5 >= i and  5 >= j      : my_ang[i][j] = 1
            if 225-22.5 < angle <= 225+22.5 :                   # 225°
                if  5 >= i and  5 <= j <= 10: my_ang[i][j] = 1
            if 270-22.5 < angle <= 270+22.5 :                   # 270°
                if  5 >= i and  10 <= j     : my_ang[i][j] = 1
            if 315-22.5 < angle <= 315+22.5 :                   # 315°
                if  5 <= i <=10 and 10 <= j : my_ang[i][j] = 1
    #print(my_ang)
    return my_ang

def fill_score(np_sco, x, y):
    for i in range(x, x+4):
        for j in range(y, y+4):
            np_sco[i, j] = 1
    
# 得点ベクトルを返す
def get_sco_matrix(score, point):
    np_sco = np.zeros([16, 16])
    if score[8]  == point : fill_score(np_sco, 12,  4)   #  8:Tomato_N
    if score[9]  == point : fill_score(np_sco, 12,  8)   #  9:Tomato_S
    if score[10] == point : fill_score(np_sco,  8,  0)   # 10:Omelette_N
    if score[11] == point : fill_score(np_sco,  4,  0)   # 11:Omelette_S
    if score[12] == point : fill_score(np_sco,  8, 12)   # 12:Pudding_N
    if score[13] == point : fill_score(np_sco,  4, 12)   # 13:Pudding_S
    if score[14] == point : fill_score(np_sco,  0,  4)   # 14:OctopusWiener_N
    if score[15] == point : fill_score(np_sco,  0,  8)   # 15:OctopusWiener_S
    if score[16] == point : fill_score(np_sco,  8,  4)   # 16:FriedShrimp_N
    if score[17] == point : fill_score(np_sco,  8,  8)   # 17:FriedShrimp_E
    if score[18] == point : fill_score(np_sco,  4,  4)   # 18:FriedShrimp_W
    if score[19] == point : fill_score(np_sco,  4,  8)   # 19:FriedShrimp_S
    return np_sco

# 自分の側面と背面の得点
def get_side_matrix(side1, side2, back):
    np_sco = np.zeros([16, 16])
    for i in range(16):
        for j in range(16):
            if not side1 == 0 :
                if 7 >= i :
                    if 8 <= j : np_sco[i][j] = 1
            if not side2 == 0 :
                if 8 <= i :
                    if 8 <= j : np_sco[i][j] = 1
            if not back  == 0 :
                if 7 >= j :     np_sco[i][j] = 1
            #if not side1 == 0 :
            #    if 7 >= i : np_sco[i][j] = 1
            #if not side2 == 0 :
            #    if 8 <= i : np_sco[i][j] = 1
    return np_sco

# gazebo座標からamcl_pose座標に変換する
def convert_coord_from_gazebo_to_amcl(my_color, gazebo_x, gazebo_y):
    if my_color == 'r':
        amcl_x    =  gazebo_y
        amcl_y    = -gazebo_x
    else:
        amcl_x    = -gazebo_y
        amcl_y    =  gazebo_x
    return amcl_x, amcl_y

class RandomBot():

    # 現在の状態を取得
    def getState(self):
        
        # 位置情報
        my_angle = quaternion_to_euler(Quaternion(self.pos[2], self.pos[3], self.pos[4], self.pos[5]))
        #my_pos = get_pos_matrix(self.pos[0], self.pos[1])                      # 自分の位置
        self.my_pos = get_pos_matrix(self.pos[0], self.pos[1]) + 0.5*self.my_pos  # 自分の位置(軌跡対応)
        self.my_pos = np.clip(self.my_pos, 0, 1)                                  # 自分の位置(軌跡対応)
        my_pos = self.my_pos                                                      # 自分の位置(軌跡対応)
        #en_pos = get_pos_matrix(self.pos[6], self.pos[7])  # 相手の位置
        self.en_pos = get_pos_matrix(self.pos[6], self.pos[7]) + 0.5*self.en_pos # 相手の位置(軌跡対応)
        self.en_pos = np.clip(self.en_pos, 0, 1)                                 # 相手の位置(軌跡対応)
        en_pos = self.en_pos                                                     # 相手の位置(軌跡対応)
        my_ang = get_ang_matrix(my_angle.z)                                    # 自分の向き
        
        # 審判情報の更新(点数)
        rospy.Subscriber("war_state", String, self.callback_war_state, queue_size=10)
        my_sco      = get_sco_matrix(self.score,  1)                               # 自分の点数
        en_sco      = get_sco_matrix(self.score, -1)                               # 相手の点数
        mySide_sco  = get_side_matrix(self.score[6], self.score[7], self.score[8]) # 自分側面と背面の点数
        enSide_sco  = get_side_matrix(self.score[3], self.score[4], self.score[5]) # 相手側面と背面の点数

        # 状態と報酬の更新( 16 × 16 × 7ch )
        state       = np.concatenate([np.expand_dims(my_pos,     axis=2),
                                     np.expand_dims(en_pos,     axis=2),
                                     np.expand_dims(my_ang,     axis=2),
                                     np.expand_dims(my_sco,     axis=2),
                                     np.expand_dims(en_sco,     axis=2),
                                     np.expand_dims(mySide_sco, axis=2),
                                     np.expand_dims(enSide_sco, axis=2)], axis=2)
        state       = np.reshape(state, [1, 16, 16, 7])                         # 現在の状態(自分と相手の位置、点数)
        
        return state

    # クラス生成時に最初に呼ばれる
    def __init__(self, bot_name, color='r', Sim_flag=True):
        self.name     = bot_name                                        # bot name 
        self.vel_pub  = rospy.Publisher('cmd_vel', Twist, queue_size=1) # velocity publisher
        self.sta_pub  = rospy.Publisher("/gazebo/model_states", ModelStates, latch=True) # 初期化用
        self.timer    = 0                                               # 対戦時間
        self.time     = 0.0                                             # 対戦時間(審判から取得)
        self.reward   = 0                                               # 報酬
        self.my_pos   = np.zeros([16, 16])     # My Location
        self.en_pos   = np.zeros([16, 16])     # En Location
        self.my_color = color                                           # 自分の色情報
        self.en_color = 'b' if color=='r' else 'r'                      # 相手の色情報
        self.score    = np.zeros(20)                                    # スコア情報(以下詳細)
        self.sim_flag = Sim_flag
         #  0:自分のスコア, 1:相手のスコア
         #  2:相手後ろ, 3:相手Ｌ, 4:相手Ｒ, 5:自分後ろ, 6:自分Ｌ, 7:自分Ｒ
         #  8:Tomato_N, 9:Tomato_S, 10:Omelette_N, 11:Omelette_S, 12:Pudding_N, 13:Pudding_S
         # 14:OctopusWiener_N, 15:OctopusWiener_S, 16:FriedShrimp_N, 17:FriedShrimp_E, 18:FriedShrimp_W, 19:FriedShrimp_S
        self.pos      = np.zeros(12)                                    # 位置情報(以下詳細)
         #  0:自分位置_x,  1:自分位置_y,  2:自分角度_x,  3:自分角度_y,  4:自分角度_z,  5:自分角度_w
         #  6:相手位置_x,  7:相手位置_y,  8:相手角度_x,  9:相手角度_y, 10:相手角度_z, 11:相手角度_w
        self.w_name = "imageview-" + self.my_color
        # cv2.namedWindow(self.w_name, cv2.WINDOW_NORMAL)
        # cv2.moveWindow(self.w_name, 100, 100)
        camera_resource_name = 'image_raw' if self.my_color == 'r' else 'image_raw'
        self.image_pub = rospy.Publisher(camera_resource_name, Image, queue_size=10)
        self.img = None
        self.debug_preview = False
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(camera_resource_name, Image, self.imageCallback, queue_size=10)
        self.debug_log_fname = None
        #self.debug_log_fname = 'log-' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '-' + self.my_color + '.csv'
        self.training = True and self.sim_flag
        self.debug_use_gazebo_my_pos = False
        self.debug_use_gazebo_enemy_pos = False
        self.debug_gazebo_my_x = np.nan
        self.debug_gazebo_my_y = np.nan
        self.debug_gazebo_enemy_x = np.nan
        self.debug_gazebo_enemy_y = np.nan
        if self.debug_use_gazebo_my_pos is False:
            if self.my_color == 'r' : rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl_pose)
            if self.my_color == 'b' : rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.callback_amcl_pose)
        if self.debug_use_gazebo_enemy_pos is False:
            self.pos[6] = 1.3 if self.my_color == 'r' else -1.3
            self.pos[7] = 0
        if (self.debug_use_gazebo_my_pos is True) or (self.debug_use_gazebo_enemy_pos is True) or (self.debug_log_fname is not None):
            rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback_model_state, queue_size=10)
        if self.debug_log_fname is not None:
            with open(self.debug_log_fname, mode='a') as f:
                f.write('my_x,my_y,my_qx,my_qy,my_qz,my_qw,my_ax,my_ay,my_az,enemy_x,enemy_y,enemy_qx,enemy_qy,enemy_qz,enemy_qw,enemy_ax,enemy_ay,enemy_az,circle_x,circle_y,circle_r,est_enemy_x,est_enemy_y,est_enemy_u,est_enemy_v,est_enemy_theta,gazebo_my_x,gazebo_my_y,gazebo_enemy_x,gazebo_enemy_y,diff_my_x,diff_my_y,diff_enemy_x,diff_enemy_y\n')
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction) # RESPECT @seigot]
        
        # 初期状態を取得
        #self.state = np.zeros([1, 16, 16, 7])                        # 状態
        self.state = self.getState()
        
        self.action = np.array([0, 0])
        self.action2 = np.array([0, 0])
        self.same_action_count = 0

        # Check if current path is valid or not in callback_global_plan()
        self.is_valid_plan = False
        rospy.Subscriber("move_base/DWAPlannerROS/global_plan", Path, self.callback_global_plan, queue_size=2)

    # スコア情報の更新(war_stateのコールバック関数)
    def callback_war_state(self, data):
        json_dict = json.loads(data.data)                  # json辞書型に変換
        self.time = json_dict['time']
        self.score[0] = json_dict['scores'][self.my_color] # 自分のスコア
        self.score[1] = json_dict['scores'][self.en_color] # 相手のスコア
        if json_dict['state'] == 'running':
            try:
                for i in range(18):
                    player = json_dict['targets'][i]['player']
                    if player == self.my_color : self.score[2+i] =  float(json_dict['targets'][i]['point'])
                    if player == self.en_color : self.score[2+i] = -float(json_dict['targets'][i]['point'])
                if self.my_color == 'b':                           # 自分が青色だった場合、相手と自分を入れ替える
                    for i in range(3) : self.score[2+i], self.score[5+i] = self.score[5+i], self.score[2+i]
            except:
                #print('callback_war_state: Invalid input ' + e)
                print('callback_war_state: Invalid input ')

    # 位置情報の更新(amcl_poseのコールバック関数)
    def callback_amcl_pose(self, data):
        pos = data.pose.pose.position
        ori = data.pose.pose.orientation
        self.pos[0] = pos.x; self.pos[1] = pos.y; self.pos[2] = ori.x; self.pos[3] = ori.y; self.pos[4] = ori.z; self.pos[5] = ori.w

    # Callback for topic ~/move_base/DWAPlannerROS/global_plan
    def callback_global_plan(self, path):
        self.is_valid_plan = True
    
    # 位置情報の更新(model_stateのコールバック関数)
    def callback_model_state(self, data):
        #print('*********', len(data.pose))
        if 'red_bot' in data.name:
            index_r = data.name.index('red_bot')
        else:
            print('callback_model_state: red_bot not found')
            return
        if 'blue_bot' in data.name:
            index_b = data.name.index('blue_bot')
        else:
            print('callback_model_state: blue_bot not found')
            return
        #print('callback_model_state: index_r=', index_r, 'index_b=', index_b)
        my    = index_r if self.my_color == 'r' else index_b
        enemy = index_b if self.my_color == 'r' else index_r
        gazebo_my_x,    gazebo_my_y    = convert_coord_from_gazebo_to_amcl(self.my_color, data.pose[my   ].position.x, data.pose[my   ].position.y)
        gazebo_enemy_x, gazebo_enemy_y = convert_coord_from_gazebo_to_amcl(self.my_color, data.pose[enemy].position.x, data.pose[enemy].position.y)
        if self.debug_use_gazebo_my_pos is True:
            self.pos[0] = gazebo_my_x
            self.pos[1] = gazebo_my_y
            ori = data.pose[my].orientation; self.pos[2] = ori.x; self.pos[3] = ori.y; self.pos[4]  = ori.z; self.pos[5]  = ori.w
        if self.debug_use_gazebo_enemy_pos is True:
            self.pos[6] = gazebo_enemy_x
            self.pos[7] = gazebo_enemy_y
            ori = data.pose[enemy].orientation; self.pos[8] = ori.x; self.pos[9] = ori.y; self.pos[10] = ori.z; self.pos[11] = ori.w
        self.debug_gazebo_my_x    = gazebo_my_x
        self.debug_gazebo_my_y    = gazebo_my_y
        self.debug_gazebo_enemy_x = gazebo_enemy_x
        self.debug_gazebo_enemy_y = gazebo_enemy_y

    # 報酬の計算
    def calc_reward(self):
        reward = 0
        #if self.timer > turnEnd:
        if self.time > TimeLimit:
            # 試合終了(Time out)
            if self.score[0] > self.score[1]:
                reward =  1
            else:
                reward = -1
        else:
            # 試合終了(Called game)
            if self.score[0] - self.score[1] >= 10 : reward =  1  # Win
            if self.score[1] - self.score[0] >= 10 : reward = -1  # Lose
        return reward


    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ 行動計算のメイン部F
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def calcTwist(self):
        
        self.timer += 1

        # Iterate until valid plan is confirmed in case of random action (No iteration for predicted action)
        force_random_action = False   # Flag to force random action for 2nd and more trials. False for 1st trial.
        avoid_best_action = (self.same_action_count >= maxSameGoalCount)
        if avoid_best_action:
            rospy.logwarn('Avoid choosing best action, choose 2nd to 5th best instead.');
        for goal_itr_cnt in range(maxGoalItrCount):
            predicted = False
            if self.timer > 6:
                #tmp = np.transpose(self.state, (0, 3, 1, 2))
                #for i in range(0, 7) : print(i, tmp[0][i])
    
                # Get action (predicted = True if predicted, otherwise random selection)
                if not self.flag_ThreadEnd :
                    self.thread.join()
                    self.flag_ThreadEnd = True
                action, predicted = self.actor.get_action(self.state, self.timer, self.mainQN, self.my_color, self.action, self.action2, self.score[0]-self.score[1], self.sim_flag, force_random_action, avoid_best_action)
                # 移動先と角度  (中心位置をずらした後に45度反時計周りに回転)
                desti   = get_destination(action)
                yaw = np.arctan2( (desti[1]-self.pos[1]), (desti[0]-self.pos[0]) )      # 移動先の角度
            else:
                action = np.array([ 0,  0])
                if self.timer <= 2 :
                    desti = np.array([ -0.60,  0.00])
                    yaw  = 0
                elif self.timer <= 6 :
                    desti = np.array([  0.00,  0.55])
                    yaw  = -90
            
            #print('****Action****', self.timer, action, desti, yaw*360/np.pi)
            print(self.my_color, '* Action * Time=%2d : %4.2f,  Score=(%2d,%2d), Position=(%4.2f, %4.2f),  Destination=(%4.2f, %4.2f, %4.0f[deg])' % (self.timer, self.time, self.score[0], self.score[1], self.pos[0], self.pos[1], desti[0], desti[1], yaw*360/np.pi))
            print('')
            
            # Actionに従った行動  目的地の設定 (X, Y, Yaw)
            is_valid_goal = self.setGoal(desti[0], desti[1], yaw, predicted)  # Returns True if valid goal is set

            # Print messages about goal
            if is_valid_goal:
                rospy.loginfo('[%d/%d] Valid goal is set by %s.' % (goal_itr_cnt + 1, maxGoalItrCount, 'prediction' if predicted else 'random selection'))
            else:
                rospy.logerr('[%d/%d] Invalid goal is set by %s. Retrying...' % (goal_itr_cnt + 1, maxGoalItrCount, 'prediction' if predicted else 'random selection'))

            # Judge whether to break the iteration or not. Break in case that:
            # 1) goal is predicted by DQN,
            # 2) random goal is valid.
            if is_valid_goal or predicted:
                # Successfully goal is set. Break the iteration.
                break

            # Here goal is random and invalid. Need to try again. Note that random action is forced next time.
            if not predicted:
                force_random_action = True

            #self.restart()  # ******* 強制Restart用 *******
        
        # Action後の状態と報酬を取得
        next_state = self.getState()                                            # Action後の状態
        reward     = self.calc_reward()                                         # Actionの結果の報酬
        if abs(reward) == 1 : next_state = np.zeros([1, 16, 16, 7])             # 試合終了時は次の状態はない
        
        self.action2 = self.action
        self.action = action
        
        # Count up if sequentially same action is chosen
        if (self.action == self.action2).all():
            self.same_action_count += 1
            rospy.logwarn('Same goal is set for %d times (approved max %d times)' % (self.same_action_count, maxSameGoalCount))
        else:
            self.same_action_count = 0

        if self.timer > 6:
            # メモリの更新する
            self.memory.add((self.state, action, reward, next_state))               # メモリの更新する
            self.state  = next_state                                                # 状態更新
        
            # Qネットワークの重みを学習・更新する replay
            if self.training == True : learn = 1
            else                     : learn = 0
            if self.my_color == 'b'  : learn = 0
            batch_size = 40   # Q-networkを更新するバッチの大きさ
            gamma = 0.97      # 割引係数
            if (batch_size >= 2 and self.memory.len() > batch_size) and learn:
                #print('call replay timer=', self.timer)
                self.mainQN.replay(self.memory, batch_size, gamma, self.targetQN, self.my_color)
            self.targetQN.model.set_weights(self.mainQN.model.get_weights())
        
        sys.stdout.flush()
        self.reward = reward
        
        return Twist()


    # シュミレーション再開
    def restart(self):
        self.vel_pub.publish(Twist()) # 動きを止める
        self.memory.reset()
        self.score  = np.zeros(20)
        self.timer  = 0
        self.time   = 0.0
        self.reward = 0
        self.my_pos   = np.zeros([16, 16])     # My Location
        self.en_pos   = np.zeros([16, 16])     # En Location
        subprocess.call('bash ../catkin_ws/src/burger_war/burger_war/scripts/reset_state.sh', shell=True)
        #r.sleep()


    # RESPECT @seigot
    # do following command first.
    #   $ roslaunch burger_navigation multi_robot_navigation_run.launch
    # returns True if suceeded, otherwise failed (need to be set goal again)
    def setGoal(self,x,y,yaw,force_wait):
        self.client.wait_for_server()
        #print('setGoal x=', x, 'y=', y, 'yaw=', yaw)

        goal = MoveBaseGoal()
        name = 'red_bot' if self.my_color == 'r' else 'blue_bot'
        #goal.target_pose.header.frame_id = name + '/map' if self.sim_flag == True else 'map'
        goal.target_pose.header.frame_id = "map"

        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        # Euler to Quartanion
        q=tf.transformations.quaternion_from_euler(0,0,yaw)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        # Stateの戻り値詳細：PENDING, ACTIVE, RECALLED, REJECTED, PREEMPTED, ABORTED, SUCCEEDED, LOST
        #  https://docs.ros.org/diamondback/api/actionlib/html/classactionlib_1_1SimpleClientGoalState.html#a91066f14351d31404a2179da02c518a0a2f87385336ac64df093b7ea61c76fafe
        self.client.send_goal(goal)

        # Wait 1 second to check path planner topic subscription. Need to be set flag false before wait
        self.is_valid_plan = False
        self.client.wait_for_result(rospy.Duration(1))
        if not self.is_valid_plan and not force_wait:
            # No new amcl_pose message received, which means planned path is invalid.
            self.client.cancel_goal()
            return False  # Failed

        # Continue waiting
        if not self.client.wait_for_result(rospy.Duration(1)):
            self.client.cancel_goal()
        #print(self.my_color, "state=", self.client.get_state())  # 2: on the way to goal, 3: reached goal

        return True  # Succeeded


    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ スタート時に時間が掛かるので、別スレッドを立ててModel読み込みを行う
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def Read_DNN_Model(self, learning_rate, memory_size):
        global graph
        with graph.as_default():
            self.mainQN   = DQN.QNetwork(learning_rate=learning_rate)   # メインのQネットワーク
            self.targetQN = DQN.QNetwork(learning_rate=learning_rate)   # 価値を計算するQネットワーク
            self.memory   = DQN.Memory(max_size=memory_size)
            self.actor    = DQN.Actor()
        
            # 重みの読み込み
            if self.sim_flag == True :
                try:
                    self.mainQN.model.load_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')     # 重みの読み込み
                except:
                    print('No weight file found. Train from scratch')
            else                     :
                self.mainQN.model.load_weights('../wss/Yoshihama0901_ws/src/burger_war/burger_war/scripts/weight.hdf5')     # 重みの読み込み
            self.targetQN.model.set_weights(self.mainQN.model.get_weights())


    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ 戦略部(繰り返し処理を行わせる)
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def strategy(self):
        
        rospy_Rate = timeScale
        r = rospy.Rate(rospy_Rate) # １秒間に送る送信回数 (change speed 1fps)
        
        # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
        learning_rate = 0.0005          # Q-networkの学習係数
        memory_size   = 1000             # バッファーメモリの大きさ
        #self.Read_DNN_Model(learning_rate, memory_size)
        self.flag_ThreadEnd = False
        self.thread = threading.Thread(target=self.Read_DNN_Model, args=([learning_rate, memory_size]), name='Read_DNN_Model_Thread')
        self.thread.start()
        #self.thread.join()

        while not rospy.is_shutdown():
            
            twist = self.calcTwist()    # 移動距離と角度を計算
            #self.vel_pub.publish(twist) # ROSに反映
            
            if self.training == True:
                # 試合終了した場合
                if self.my_color == 'r':
                    print('me', self.score[0], 'enemy', self.score[1], 'reward', self.reward)
                    if abs(self.reward) == 1:
                        if   self.reward == 0 : print('Draw')
                        elif self.reward == 1 : print('Win!')
                        else                  : print('Lose')
                        with open('result.csv', 'a') as f:
                            writer = csv.writer(f, lineterminator='\n')
                            writer.writerow([self.score[0], self.score[1], time.time()])
                        self.mainQN.model.save_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')            # モデルの保存
                        self.restart()                                          # 試合再開
                        r.sleep()
                else:
                    #if self.timer % turnEnd == 0 :
                    if self.time < 10 :
                        self.memory.reset()
                        # 重みの読み込み
                        while True:
                            try:
                                self.mainQN.model.load_weights('../catkin_ws/src/burger_war/burger_war/scripts/weight.hdf5')
                                break
                            except:
                                print(self.my_color, 'load_weights error. Retry')
                                time.sleep(1)
                        print(self.my_color, 'load_weights OK')
            
            r.sleep()
        
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    # _/ カメラ画像読込み部分(多分変える事はないと思う)
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    def imageCallback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV_FULL)
        hsv_h = hsv[:, :, 0]
        hsv_s = hsv[:, :, 1]
        mask = np.zeros(hsv_h.shape, dtype=np.uint8)
        mask[((hsv_h < 16) | (hsv_h > 240)) & (hsv_s > 64)] = 255
        red = cv2.bitwise_and(self.img, self.img, mask=mask)
        height = self.img.shape[0]
        canny_param = 100
        canny = cv2.Canny(red, canny_param/2, canny_param)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT,
                                   dp=1, minDist=height/10, param1=canny_param, param2=8,
                                   minRadius=height/96, maxRadius=height/12)
        circle_x = -1
        circle_y = -1
        circle_r = -1
        est_enemy_x = np.nan
        est_enemy_y = np.nan
        est_enemy_u = np.nan
        est_enemy_v = np.nan
        est_enemy_theta = np.nan
        my_x = self.pos[0]
        my_y = self.pos[1]
        my_qx = self.pos[2]
        my_qy = self.pos[3]
        my_qz = self.pos[4]
        my_qw = self.pos[5]
        my_angle = quaternion_to_euler(Quaternion(my_qx, my_qy, my_qz, my_qw))
        if circles is not None:
            for i in circles[0,:]:
                x = int(i[0])
                y = int(i[1])
                r = int(i[2])
                if (y < height * 5 / 8) and (r > circle_r):
                    circle_x = x
                    circle_y = y
                    circle_r = r
            if circle_r > 0:
                est_enemy_sin_theta = -0.00143584 * circle_x \
                                      + 0.4458366274811388
                est_enemy_theta = np.rad2deg(np.arcsin(est_enemy_sin_theta))
                est_enemy_v = 4.58779425e-09 * np.power(circle_y, 4) \
                              - 1.14983273e-06 * np.power(circle_y, 3) \
                              + 1.21335973e-04 * np.power(circle_y, 2) \
                              - 7.94065667e-04 * circle_y \
                              + 0.5704722921109504
                est_enemy_u = -est_enemy_v * np.tan(np.deg2rad(est_enemy_theta))
                est_p = np.cos(np.deg2rad(my_angle.z)) * est_enemy_u \
                        - np.sin(np.deg2rad(my_angle.z)) * est_enemy_v
                est_q = np.sin(np.deg2rad(my_angle.z)) * est_enemy_u \
                        + np.cos(np.deg2rad(my_angle.z)) * est_enemy_v
                est_dx = est_q
                est_dy = -est_p
                est_enemy_x = my_x + est_dx
                est_enemy_y = my_y + est_dy
        if self.debug_use_gazebo_enemy_pos is False:
            if (not np.isnan(est_enemy_x)) and (not np.isnan(est_enemy_y)):
                self.pos[6] = est_enemy_x
                self.pos[7] = est_enemy_y
        if self.debug_log_fname is not None:
            with open(self.debug_log_fname, mode='a') as f:
                # pos[6] ... pos[11] are filled in callback_model_state
                enemy_x = self.pos[6]
                enemy_y = self.pos[7]
                enemy_qx = self.pos[8]
                enemy_qy = self.pos[9]
                enemy_qz = self.pos[10]
                enemy_qw = self.pos[11]
                enemy_angle = quaternion_to_euler(Quaternion(enemy_qx, enemy_qy, enemy_qz, enemy_qw))
                f.write('%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'
                        % (my_x, my_y, my_qx, my_qy, my_qz, my_qw,
                           my_angle.x, my_angle.y, my_angle.z,
                           enemy_x, enemy_y, enemy_qx, enemy_qy, enemy_qz, enemy_qw,
                           enemy_angle.x, enemy_angle.y, enemy_angle.z,
                           circle_x, circle_y, circle_r,
                           est_enemy_x, est_enemy_y, est_enemy_u, est_enemy_v, est_enemy_theta,
                           self.debug_gazebo_my_x, self.debug_gazebo_my_y, self.debug_gazebo_enemy_x, self.debug_gazebo_enemy_y,
                           my_x - self.debug_gazebo_my_x, my_y - self.debug_gazebo_my_y,
                           est_enemy_x - self.debug_gazebo_enemy_x, est_enemy_y - self.debug_gazebo_enemy_y))
        if self.debug_preview:
            hough = self.img.copy()
            if circles is not None:
                for i in circles[0,:]:
                    color = (255, 255, 0)
                    pen_width = 2
                    if circle_x == int(i[0]) and circle_y == int(i[1]):
                        color = (0, 255, 0)
                        pen_width = 4
                        cv2.circle(hough, (int(i[0]), int(i[1])), int(i[2]), color, pen_width)
            #cv2.imshow("red", red)
            #cv2.imshow("canny", canny)
            cv2.imshow(self.w_name, hough)
            cv2.waitKey(1)

if __name__ == '__main__':
    
    # sim環境用のフラグ。本番(実機動作)では、
    #   ・リセット動作を行わない
    #   ・学習を行わない
    #   ・確率でのランダム動作を行わない
    Sim_flag = True
    
    try:
        rside = rosparam.get_param('enemyRun/rside')
    except:
        rside = 'r'
    print('**************** rside=', rside)
    
    rospy.init_node('IntegAI_run')    # 初期化宣言 : このソフトウェアは"IntegAI_run"という名前
    bot = RandomBot('Team Integ AI', color=rside, Sim_flag=Sim_flag)
    
    bot.strategy()

