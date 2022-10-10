import numpy as np

class Env():
    def __init__(self):
        self.player = player()
        self.ball = ball()
        self.reward = 0
        self.kick = False

    def step(self,action):
        self.kick = False
        self.reward = 0
        leg = max(action[1],action[2])
        if leg > 0.8:
            if action[1] > action[2]:
                action[1],action[2] = 1,0
            else:
                action[1],action[2] = 0,1
        else:
            action[1],action[2] = 0,0
        self.player.step(action)
        self.ball.step()
        if self.kickdetection():
            self.ball.vx,self.ball.vy = self.kicked(action[1:5])

        if self.headcollision():
            self.ball.vx,self.ball.vy = self.headbound()
        
        self.reward += self.positionreward(self.ball.x,self.player.x)
        
        return [self.ball.x-self.player.x,self.ball.y,self.ball.vx,self.ball.vy],self.reward,self.kick
    
    def positionreward(self,ball_x,player_x):
        r = np.exp(-0.5*(ball_x-player_x)**2*0.5)
        return r

    def reset(self):
        self.player.reset()
        self.ball.reset()
        return self.ball.x-self.player.x,self.ball.y,self.ball.vx,self.ball.vy
    
    def kickdetection(self):
        leg_len = self.player.leg_len
        if (self.ball.y-leg_len*np.cos(np.pi/6))**2 + (self.ball.x-self.player.x)**2 <= self.player.leg_len**2 + self.ball.r**2:
            return True
        else:
            return False
    
    def kicked(self,action):
        '''
        action = [bool(left),bool(right),float(left_force),float(right_force)]
        forceはクリップする
        出力は当たる時の足の角度、左か右か、当たらない時はFalseとnp.pi/6
        '''
        leg_len = self.player.leg_len
        body_x = self.player.x
        body_y = leg_len*np.cos(np.pi/6)
        ball_x,ball_y,ball_vx,ball_vy = self.ball.x,self.ball.y,self.ball.vx,self.ball.vy
        ball_r = self.ball.r

        A = - ball_x**2 + 2*ball_x*body_x - body_x**2 + ball_r**2 
        B = ball_x*ball_y + body_x*body_y - body_x*ball_y - body_y*ball_x
        C = - ball_y**2 - body_y**2 + 2*ball_y*body_y + ball_r**2
    
        if (body_x-ball_x)**2 + (body_y-ball_y)**2 <= ball_r**2:
            return self.ball.vx,self.ball.vy

        if A != 0:
            a1 = (-B+np.sqrt((B**2)-A*C))/A
            a2 = (-B-np.sqrt((B**2)-A*C))/A
            if action[0]: #左足
                a1 = max(a1,a2)
                leg_theta = np.arctan(a1)
                leg_theta = leg_theta - np.pi/2 
                if -np.pi*3/4 <= leg_theta and leg_theta < -np.pi/6: 
                    self.player.left_theta = leg_theta
                    self.ball.ax += action[2]*np.cos(np.pi+leg_theta)
                    self.ball.ay += action[2]*np.sin(np.pi+leg_theta)
                    # print(np.pi+leg_theta)
                    # if abs(np.sin(np.pi+leg_theta)) > abs(np.cos(np.pi+leg_theta)) and np.sin(np.pi+leg_theta)>0:
                    self.reward += 100
                    print(100)
                    self.kick = True
                    return self.ball.ax,self.ball.ay
                else:
                    self.reward -= 1
                    return self.ball.vx,self.ball.vy
            elif action[1]: #右足
                a1 = min(a1,a2)
                leg_theta = np.arctan(a1)
                leg_theta = leg_theta + np.pi/2
                if np.pi/6 < leg_theta and leg_theta <= np.pi*3/4: 
                    self.player.right_theta = leg_theta
                    self.ball.ax += action[3]*np.cos(leg_theta)
                    self.ball.ay += action[3]*np.sin(leg_theta)
                    # if abs(np.sin(leg_theta)) > abs(np.cos(leg_theta)) and np.sin(leg_theta)>0:
                    self.reward += 100
                    print(100)
                    self.kick = True
                    return self.ball.ax,self.ball.ay
                else:
                    self.reward -= 1
                    return self.ball.vx,self.ball.vy
        return self.ball.vx,self.ball.vy

        '''
        足からの直線がボールに対しての接線となる時の傾きは
        a = {(b-d)^2-r^2}/{2cd-2cb-c^2+r^2}
        where
        足の式:y = ax + b
        ボールの式:(x-c)^2 + (y-d)^2 = r^2
        '''

    def headbound(self):
        head_x = self.player.x
        head_y = self.player.leg_len*np.cos(np.pi/6)+self.player.head_r+self.player.body_len
        head_r = self.player.head_r
        ball_x,ball_y,ball_vx,ball_vy = self.ball.x,self.ball.y,self.ball.vx,self.ball.vy
        ball_r = self.ball.r
        
        resi_x,resi_y = ball_x-head_x,ball_y-head_y
        if resi_x > 0:
            axis_theta = np.arctan(resi_y/resi_x)
        elif resi_x < 0:
            if resi_y >= 0:
                axis_theta = np.arctan(resi_y/resi_x) + np.pi
            else:
                axis_theta = np.arctan(resi_y/resi_x) - np.pi
        else:
            axis_theta = resi_y/np.abs(resi_y)*np.pi/2
        #  -np.pi < axis_theta <= np.pi

        if ball_vx > 0:
            if ball_vy >= 0:
                tra_theta = np.pi + np.arctan(ball_vy/ball_vx) - np.pi
            else:
                tra_theta = np.pi + np.arctan(ball_vy/ball_vx) + np.pi
        elif ball_x < 0:
            tra_theta = np.arctan(ball_vy/ball_vx)
        else:
            tra_theta = ball_vy/np.abs(ball_vy)*np.pi/2
        #  -np.pi < tra_theta <= np.pi

        resi_theta = np.abs(np.abs(axis_theta)-np.abs(tra_theta))
        if resi_theta >= np.pi:
            resi_theta = np.pi - resi_theta
        
        if np.abs(tra_theta) >= np.abs(axis_theta):
            theta = axis_theta - axis_theta/np.abs(axis_theta)*resi_theta
        else:
            theta = axis_theta + axis_theta/np.abs(axis_theta)*resi_theta

        # if abs(np.cos(theta)) < 1/2 and np.sin(theta)>0:
        self.reward += 20
        print(20)

        # ballの座標を一度当たったらheadとの距離がhead_r**2+ball_r**2になるようにしておく
        if 0 <= theta and theta < np.pi/2:
            bound_vx = np.sqrt((ball_vx**2+ball_vy**2)/(1+np.tan(theta)**2))
            bound_vy = np.tan(theta)*bound_vx
            return bound_vx,bound_vy
        elif np.pi/2 < theta:
            bound_vx = -np.sqrt((ball_vx**2+ball_vy**2)/(1+np.tan(theta)**2))
            bound_vy = np.tan(theta)*bound_vx
            return bound_vx,bound_vy
        elif theta < -np.pi/2:
            bound_vx = np.sqrt((ball_vx**2+ball_vy**2)/(1+np.tan(theta)**2))
            bound_vy = np.tan(theta)*bound_vx
            return bound_vx,bound_vy
        elif -np.pi/2 < theta and theta < 0:
            bound_vx = np.sqrt((ball_vx**2+ball_vy**2)/(1+np.tan(theta)**2))
            bound_vy = np.tan(theta)*bound_vx
            return bound_vx,bound_vy
        else:
            bound_vx = 0
            bound_vy = np.sin(theta)*np.sqrt(ball_vx**2+ball_vy**2)
            return bound_vx,bound_vy

    def headcollision(self):
        leg_len = self.player.leg_len
        body_len = self.player.body_len
        head_r = self.player.head_r
        if (self.ball.y-(leg_len*np.cos(np.pi/6)+body_len+head_r))**2 + (self.ball.x-self.player.x)**2 <= self.player.head_r**2 + self.ball.r**2:
            self.reward += 10
            return True
        else:
            return False

    def drawing(self,ax):
        def player(ax,x):
            left_theta = self.player.left_theta
            right_theta = self.player.right_theta
            head_r = self.player.head_r
            leg_len = self.player.leg_len
            body_len = self.player.body_len
            head_x_arr = []
            head_y_arr = []
            for i in range(31):
                head_x_arr.append(x+np.cos(2*np.pi/30*i)*head_r)
                head_y_arr.append(leg_len*np.cos(np.pi/6)+body_len+head_r+np.sin(2*np.pi/30*i)*head_r)
            head, = ax.plot(head_x_arr,head_y_arr,color='b')
            body_leg_x_arr = [x+leg_len*np.sin(left_theta),x,x,x,x+leg_len*np.sin(right_theta)]
            body_leg_y_arr = [leg_len*np.cos(np.pi/6)-leg_len*np.cos(left_theta),leg_len*np.cos(np.pi/6),leg_len*np.cos(np.pi/6)+body_len,\
                            leg_len*np.cos(np.pi/6),leg_len*np.cos(np.pi/6)-leg_len*np.cos(right_theta)]
            body_leg, = ax.plot(body_leg_x_arr,body_leg_y_arr,color='b')
            return head,body_leg

        def ball(ax,x,y):
            ball_r = self.ball.r
            ball_x_arr = []
            ball_y_arr = []
            for i in range(31):
                ball_x_arr.append(x+np.cos(2*np.pi/30*i)*ball_r)
                ball_y_arr.append(y+np.sin(2*np.pi/30*i)*ball_r)
            ball, = ax.plot(ball_x_arr,ball_y_arr,color='r')
            return ball

        return player(ax,self.player.x),ball(ax,self.ball.x,self.ball.y)

class player():
    def __init__(self):
        self.x = 0
        self.head_r = 0.1
        self.body_len = 0.2
        self.leg_len = 0.3
        self.left_theta = -np.pi/6
        self.right_theta = np.pi/6
        self.dt = 0.01
    
    def step(self,action):
        self.x += action[0]*self.dt
        if not action[1]:
            self.left_theta = -np.pi/6
        # if action[1] < 0 and action[1] > -np.pi/4*3:
        #     self.left_theta = action[1]
        # else:
        #     self.left_theta = -np.pi/6
        if not action[2]:
            self.right_theta = np.pi/6
        # if action[2] > 0 and action[2] < np.pi/4*3:
        #     self.left_theta = action[2]
        # else:
        #     self.left_theta = np.pi/6

    def reset(self):
        self.x = 0
        self.left_theta = -np.pi/6
        self.right_theta = np.pi/6

class ball():
    def __init__(self):
        self.r = 0.1
        self.g = -9.8
        self.dt = 0.01
        self.x = 0
        self.y = 2
        self.vx = 0
        self.vy = -1
        self.ax = 0
        self.ay = 0
    
    def step(self):
        if self.wallbound():
            self.vx = -self.vx*0.6
            if self.x < -2:
                self.x = -2
            elif self.x > 2:
                self.x = 2
        if self.ceilingbound():
            self.vy = -self.vy*0.6
            if self.y > 4:
                self.y = 4
        kp = 0.1 #空気抵抗
        self.ax -= self.vx*kp
        self.ay = self.g - self.vy*kp
        self.vx += self.ax*self.dt
        self.vy += self.ay*self.dt
        self.x += self.vx*self.dt
        self.y += self.vy*self.dt
        self.ax = 0
        self.ay = self.g
    
    def reset(self):
        self.x = 0
        self.y = 2
        self.vx = 0
        self.vy = 0
        self.ax = 0
    
    def kicked(self):
        return 
    
    def wallbound(self):
        if self.x < -2 or self.x > 2:
            return True
        else:
            return False
    
    def ceilingbound(self):
        if self.y > 4:
            return True
        else:
            return False

    
    def miss(self):
        if self.y < 0:
            return True
        else:
            return False
