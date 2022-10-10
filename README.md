# liftingAI
二次元平面上でリフティングするAI

AgentはSACによって行動選択を行う。

# Environment
•スペース
xが-2,2の位置に壁があり、そこにボールが当たると跳ね返る。

yが4の位置に天井がありそこに当たると跳ね返る。

yが0を下回ると初期位置から再スタートされる。


•action space

[v(float),left(float),right(float),l_force(float),r_force(float)]

v：playerのx軸方向の速度

left：左足を振るかどうか(1 if left > 0.8 else 0)

right：右足を振るかどうか(1 if right > 0.8 else 0)

l_force：左足の力

r_force：右足の力


•observation space

[ball_x(float),player_x,ball_y(float),ball_vx(float),ball_vy(float)]

player_x：playerのx座標

ball_x：ballのx座標

ball_y：ballのy座標

ball_vx：ballのx軸方向の速度

ball_vy：ballのy軸方向の速度


•描画
matplotlib


# 結果
•episodeが10ぐらいでballから逃げるようになる。

•描画がおかしい時がある。

# 考察

•episodeが10ぐらいでballから逃げるようになる。

→task horizonであるために、報酬の設定が上手くいっていないと学習が全く収束しない。

→ballを蹴り上げたり、ヘディングすると報酬を与えているが、ballに触れていない時の報酬の与え方をもう少し工夫しないといけない

→シンプルに学習させるactionが難しい(→足を振るタイミングも学習してほしかったために、振るタイミングを決めるactionを入れているが、難しすぎるのかもしれない)


•描画がおかしい時がある。

→描画のdtをもう少し短くすれば、変な挙動はなくなるかもしれな(ただしステップ数が多くなり、余計に学習が難しくなるため、simulationのstepと分けてcodeを書く必要がある)
