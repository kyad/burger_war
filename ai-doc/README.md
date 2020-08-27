# 環境構築手順
- 基本的には公式手順 https://github.com/OneNightROBOCON/burger_war に従う
- 公式手順に加えて必要な手順
   ```
   sudo apt install ros-kinetic-dwa-local-planner
   cd ~/catkin_ws/src/burger_war
   pip install -r requirements.txt
   ```

# 起動方法
## 学習する
### 学習する
- `git log`
- `burger_war/scipits/clean.sh`
- `rm burger_war/script/weight.hdf5`  # スクラッチで学習する場合
- burger_war/launch/your_burger.launchで、sim_flagとtrainingをTrueにする
- burger_war/launch/enemy.launchで、sim_flagとtrainingをTrueにする（自分自身と戦わせる場合）
- burger_war/scirpts/TeamEmbeddedAI.pyで、realTimeFactorを適切に設定する
- `git diff`
- `bash scripts/sim_with_judge.sh`
- GazeboのReal Time Factorを設定する
- `bash scripts/start.sh -l 0`  # 自分自身と戦わせる
- `bash scripts/start.sh -l 4`  # 敵チームと戦わせる

### Lv1~Lv3と戦わせて学習する
- your_burger.launchで、sim_flagとtrainingをTrueにする
- enemy.launchで、sim_flagとtrainingをTrueにする
- `rosclean check`
- `rosclean purge`
- `rm ~/.ros/result.csv`
- `rm burger_war/script/weight.hdf5  # スクラッチで学習する場合`
- TeamEmbeddedAI.pyで、realTimeFactorを適切に設定する
- `bash scripts/sim_with_judge.sh`
- GazeboのReal Time Factorを設定する
- `bash scripts/start.sh -l (1|2|3)`

## 2つのモデルのどちらが良いか戦わせて評価する
- your_burger.launchで、model_fileに自機のモデルのパスを設定、sim_flagをTrue、trainingをFalseにする
- `rosclean check`
- `rosclean purge`
- `rm ~/.ros/result.csv`
- enemy.launchで、model_fileに敵機のモデルのパスを設定、sim_flagをTrue、trainingをFalseにする
- TeamEmbeddedAI.pyで、realTimeFactor, maxGameCountを適切に設定する
- `bash scripts/sim_with_judge.sh`
- GazeboのReal Time Factorを設定する
- `bash scripts/start.sh -l 0`
- 試合結果は~/.ros/result.csvに格納される

## 試合用の動作をする
### 自分自身と戦わせる
- your_burger.launchで、sim_flagとtrainingをFalseにする
- enemy.launchで、sim_flagとtrainingをFalseにする
- TeamEmbeddedAI.pyで、realTimeFactorを適切に設定する
- `bash scripts/sim_with_judge.sh`
- GazeboのReal Time Factorを設定する
- `bash scripts/start.sh -l 0`

### Lv1~Lv3と戦わせる
- your_burger.launchで、sim_flagとtrainingをFalseにする
- TeamEmbeddedAI.pyで、realTimeFactorを適切に設定する
- `bash scripts/sim_with_judge.sh`
- GazeboのReal Time Factorを設定する
- `bash scripts/start.sh -l (1|2|3)`

# 参考にしたURL
- https://github.com/OneNightROBOCON/burger_war
- https://qiita.com/sugulu/items/bc7c70e6658f204f85f9
- http://home.q00.itscom.net/otsuki/alphaZero.pdf
