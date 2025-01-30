import flappy_bird_gymnasium
import gymnasium

def agent(obs):
    pipe = 0
    if obs[0] < 5:
        pipe = 1
    x = obs[pipe * 3]
    bot = obs[pipe * 3 + 2]
    top = obs[pipe * 3 + 1]
    y_next = obs[-3] + obs[-2] + 24 + 1
    if 74 < x < 88 and obs[-3] - 45 >= top:
        return 1
    elif y_next >= bot:
        return 1
    return 0


def play(use_lidar=True, render_mode="human"):
    env = gymnasium.make(
        "FlappyBird-v0",
        audio_on=True,
        render_mode=render_mode,
        use_lidar=use_lidar,
        normalize_obs=False,
        score_limit=1000,
    )

    steps = 0
    video_buffer = []

    obs, _ = env.reset()
    while True:
        # Getting action:
        action = agent(obs)
        # Processing:
        obs, reward, done, term, info = env.step(action)
        video_buffer.append(obs)

        steps += 1

        if done or term:
            break

    env.close()
    return info["score"]


if __name__ == "__main__":
    play(use_lidar=False, render_mode="human")
