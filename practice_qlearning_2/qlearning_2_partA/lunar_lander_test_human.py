import time
import gymnasium as gym
import pygame


def test_human():
    # Initialize Gymnasium environment with human rendering mode
    try:
        env = gym.make("LunarLander-v3", render_mode="human", gravity=-1.620)
    except Exception:
        env = gym.make("LunarLander-v2", render_mode="human", gravity=-1.620)
    # Initialize Pygame to capture keyboard input
    pygame.init()
    # Action reference:
    # 0: Do nothing
    # 1: Fire left engine  (pushes lander RIGHT)
    # 2: Fire main engine  (pushes lander UP)
    # 3: Fire right engine (pushes lander LEFT)
    clock = pygame.time.Clock()
    FPS = 30  # Control speed of physics simulation
    state, info = env.reset()
    total_reward = 0
    running = True

    print("Controles. FLECHAS: [UP] JET PRINCIPAL | [IZQ] Motor IZQ | [DER] Motor DER")
    while running:
        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Detect currently held keys
        keys = pygame.key.get_pressed()
        # Determine action based on pressed key
        action = 0  # Default to do nothing
        if keys[pygame.K_UP]:
            action = 2  # Main thrust
        elif keys[pygame.K_LEFT]:
            action = 3  # Fire right engine to steer LEFT
        elif keys[pygame.K_RIGHT]:
            action = 1  # Fire left engine to steer RIGHT
        # Step the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode finished! Total Reward: {total_reward:.2f}")
            time.sleep(1)  # Brief pause before restarting
            state, info = env.reset()
            total_reward = 0
            clock.tick(FPS)
    env.close()
    pygame.quit()


if __name__ == "__main__":
    test_human()