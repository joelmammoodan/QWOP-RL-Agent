
import qwop_gym
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO



class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env,repeat=3):
        super().__init__(env)
        self.repeat=repeat
    
    def step(self,action):
        total_reward=0
        for _ in range(self.repeat):
            obs,reward,done,trunc,info=self.env.step(action)
            total_reward+=reward
            if done:
                break
        return obs,total_reward,done,trunc,info
    


class SpeedSurvivalReward(gym.Wrapper):

    #env:the OG QWOP environment
    #speed_weight:how much reward the agent gets
    #survival_reward:reward for not dying each step
    #stagnation_penalty:penalty for not moving
    #min_speed=minimum speed for the agent to attain to avoid penalty
    #backward_penalty_weight:penalty for moving backward
    #upright_angle_threshold:angle to be considered to be upright
    #upright_time_penalty_rate:how fast penalty increases when falling


    def __init__(
        self,
        env,
        speed_weight=1.0,
        survival_reward=0.01,
        stagnation_penalty=0.1,
        min_speed=0.01,
        backward_penalty_weight=5.0,
        upright_angle_threshold=0.35,     # radians from vertical
        upright_time_penalty_rate=0.02    # grows with time
    ):
        super().__init__(env)

        self.speed_weight = speed_weight
        self.survival_reward = survival_reward
        self.stagnation_penalty = stagnation_penalty
        self.min_speed = min_speed
        self.backward_penalty_weight = backward_penalty_weight

        self.upright_angle_threshold = upright_angle_threshold
        self.upright_time_penalty_rate = upright_time_penalty_rate

        self.prev_x = None
        self.not_upright_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = self._get_x(obs, info)
        self.not_upright_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)

        curr_x = self._get_x(obs, info)

        # ------------------------
        # Forward / backward logic
        # ------------------------
        if self.prev_x is not None and curr_x is not None:
            dx = curr_x - self.prev_x

            if dx > 0:
                reward += self.speed_weight * dx

            if dx < 0:
                reward -= self.backward_penalty_weight * abs(dx)

            if abs(dx) < self.min_speed:
                reward -= self.stagnation_penalty

        # ------------------------
        # Upright-time penalty
        # ------------------------
        upright = self._is_upright(obs, info)

        if not upright:
            self.not_upright_steps += 1
            reward -= self.upright_time_penalty_rate * self.not_upright_steps
        else:
            self.not_upright_steps = 0

        # ------------------------
        # Survival reward
        # ------------------------
        if not done:
            reward += self.survival_reward

        self.prev_x = curr_x
        return obs, reward, done, trunc, info

    # -------- helpers --------

    def _get_x(self, obs, info):
        if info is not None:
            if "x" in info:
                return info["x"]
            if "position" in info:
                return info["position"][0]

        if isinstance(obs, (list, tuple, np.ndarray)):
            return obs[0]

        return None

    def _is_upright(self, obs, info):
        """
        Returns True if torso is upright enough.
        Adjust this function if qwop_gym exposes posture differently.
        """

        # Preferred: torso angle from vertical
        if info is not None and "torso_angle" in info:
            return abs(info["torso_angle"]) < self.upright_angle_threshold

        # Fallback: assume obs[2] ~ torso angle
        if isinstance(obs, (list, tuple, np.ndarray)) and len(obs) > 2:
            return abs(obs[2]) < self.upright_angle_threshold

        # If unknown, assume upright (safe fallback)
        return True

def main():
    env = gym.make(
        "QWOP-v1",
        browser=r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        driver=r"C:\Users\LEGION\qwop_walk\chromedriver-win64\chromedriver-win64\chromedriver.exe",
        render_mode='human'
    )

    env=ActionRepeatWrapper(env)
    env=SpeedSurvivalReward(env)
    model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4, #higher the value, faster it learns ,but is very unstable
    gamma=0.99,         # care short term goals --> care for long term goals
    ent_coef=0.005,      # stuck in same methods --> very chaotic
    clip_range=0.15,    #small policy updates --> huge policy updates
    batch_size=128,     #fast coarse samples --> fine slow samples
    n_steps=2048,       #fast updates (stepwise)--> slow but stable updates
    gae_lambda=0.95,    #biasing the reward --> more accurate reward
    vf_coef=0.5,        #weak guidance by value function --> overfocusing on value function
    max_grad_norm=0.5,  #limit to stop huge updates to network
    verbose=2           #0-silent 1-reports(brief) 2-detailed report
    )

    model.learn(total_timesteps=1500_000)
    model.save("ppo_qwop")

    env.close()
    
if(__name__=="__main__"):
    main()
