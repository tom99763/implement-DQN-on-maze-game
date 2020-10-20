import gym
import gym_maze
from rl_brain import DQN


def simulate():
    for ep in range(EPOCH):
        total_reward = 0
        s = env.reset()
        while True:
            action = algo.choose_action(s)
            s_, r, t, info = env.step(action)
            algo.save_memory(s, action, r, s_)
            total_reward += r
            if algo.can_provide_sample(batch_size):
                algo.algorithm()
            if t:
                print(f'episode---{ep}, total_reward----{total_reward}')
                break
            env.render()
            s = s_


if __name__ == "__main__":
    id = 'maze-random-10x10-v0'
    env = gym.make(id)
    memory_capacity = 2500
    batch_size = 500
    hidden = 10
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    replace_iter = 100
    EPOCH = 100

    algo = DQN(state_space, action_space, hidden,
               memory_capacity, batch_size, replace_iter)

    simulate()
