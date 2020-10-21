import gym
import gym_maze
from algorithm import DQN


def simulate():
    step = 0
    env.render()
    for episode in range(EPOCH):
        print(f'episode---{episode}')
        s = env.reset()

        while True:

            env.render()

            action = RL.choose_action(s)

            s_, r, done, info = env.step(action)

            RL.save_to_memory(s, action, r, s_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            s = s_

            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.close()


if __name__ == "__main__":
    id = 'maze-sample-5x5-v0'
    env = gym.make(id)
    memory_size = 3000
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]
    replace_iter = 400
    EPOCH = 300

    RL = DQN(state_space, action_space, memory_size, replace_iter)

    simulate()

    RL.plot_cost()
