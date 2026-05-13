import qwop_gym
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env



class ActionRepeat(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


def main():
    env = gym.make(
        "QWOP-v1",
        browser=r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        driver=r"C:\Users\LEGION\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe",
        render_mode='human'
    )


    env=make_vec_env(lambda:ActionRepeat(env,repeat=2),n_envs=1)
   
    

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=20_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        verbose=1
    )

    model.learn(total_timesteps=500_000)
    model.save("ddqn_qwop")

    env.close()

    
if(__name__=="__main__"):
    main()
