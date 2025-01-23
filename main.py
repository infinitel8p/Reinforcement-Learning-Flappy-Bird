import flappy_bird_gym as fbg
import time

env = fbg.make("FlappyBird-v0")

obs = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample() # for a random action

    # Processing:
    obs, reward, done, info = env.step(action)

    print(obs)
    print(reward)
    print(done)
    print(info)
    
    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS
    
    # Checking if the player is still alive
    if done:
        break

env.close()
