
import qwop_gym
import gymnasium as gym

from stable_baselines3 import PPO


class ContinuousThighWrapper(gym.Wrapper):
    """
    Rewards QWOP agent continuously for:
    - Forward movement (dx)
    - Thigh rotation
    - Speed implicitly via forward movement per step
    Includes a small step penalty.
    """
    def __init__(self, env, distance_multiplier=5.0,
                 angle_multiplier=5.0, step_penalty=-0.1, hold_steps=3):
        super().__init__(env)
        self.distance_multiplier = distance_multiplier
        self.angle_multiplier = angle_multiplier
        self.step_penalty = step_penalty
        self.hold_steps = hold_steps

        # Action hold state
        self._hold_counter = 0
        self._last_action = None

        # Previous state
        self.prev_thigh_L = 0.0
        self.prev_thigh_R = 0.0
        self.prev_x = 0.0
        self.steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = obs[0]
        self.steps = 0
        self._hold_counter = 0
        self._last_action = None
        self.prev_thigh_L, self.prev_thigh_R = obs[1], obs[3]
        return obs, info

    def step(self, action):
        # Hold previous action for a few steps if needed
        if self._hold_counter > 0:
            use_action = self._last_action
            self._hold_counter -= 1
        else:
            use_action = action
            self._last_action = action
            self._hold_counter = self.hold_steps - 1

        obs, reward_raw, terminated, truncated, info = self.env.step(use_action)
        done = terminated or truncated
        print(obs,reward_raw,terminated,truncated)
        # Forward progress
        x = obs[0]
        dx = x - self.prev_x

        # Thigh rotation
        thigh_L, thigh_R = obs[1], obs[3]
        delta_thigh_L = abs(thigh_L - self.prev_thigh_L)
        delta_thigh_R = abs(thigh_R - self.prev_thigh_R)

        # Compute reward
        reward = (
            dx * self.distance_multiplier +                 # forward motion
            (delta_thigh_L + delta_thigh_R) * self.angle_multiplier +  # thigh movement
            self.step_penalty                               # per-step penalty
        )

        # Update previous state
        self.prev_x = x
        self.prev_thigh_L = thigh_L
        self.prev_thigh_R = thigh_R
        self.steps += 1

        return obs, reward, done, truncated, info



def main():
    env = gym.make(
        "QWOP-v1",
        browser=r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        driver=r"C:\Users\LEGION\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe",
        render_mode='human'
    )

    #env=ContinuousThighWrapper(env)
   
    model = PPO.load('ppo_qwop')


    obs, info = env.reset()
    done = False

    while not done:
        # Get the action from the model
        action, _states = model.predict(obs, deterministic=True)  # deterministic=True to see best learned moves

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
if(__name__=="__main__"):
    main()
