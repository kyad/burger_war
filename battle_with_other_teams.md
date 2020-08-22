# 他チームとの対戦（シミュレーター上）

## 他チームのコードの置き場

`burger_war_emeny`と`burger_navigation_enemy`

## VS.高田さんチーム

高田さんチームと対戦するために必要なライブラリなどのインストール([高田さんチームのrepo](https://github.com/seigot/burger_war))

```
$ sudo apt-get install ros-kinetic-global-planner
$ sudo apt install -y libarmadillo-dev libarmadillo6
$ cd ~/catkin_ws/src
$ git clone https://github.com/tysik/obstacle_detector.git
$ catkin_init_workspace
$ cd ~/catkin_ws
$ catkin_make
$ bash devel/setup.sh
```

高田さんチームとの対戦方法は
```
$ cd ~/catkin_ws/src/burger_war
$ bash scripts/start.sh -l 4
```
のように`-l 4`をつければよい。

