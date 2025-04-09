import numpy as np


class TrafficEnvironment:
    def __init__(self):
        self.state = np.random.randint(0, 10, size=4)  # [North, South, East, West]
        self.current_light = 0  # 0: Green for NS, 1: Green for EW

    def step(self, action):
        if action == 1:
            self.current_light = 1 - self.current_light  # Switch traffic lights

        if self.current_light == 0:  # NS Green, EW Red
            passed = min(self.state[0], np.random.randint(1, 5)) + min(self.state[1], np.random.randint(1, 5))
            self.state[0] = max(0, self.state[0] - passed)
            self.state[1] = max(0, self.state[1] - passed)
        else:  # EW Green, NS Red
            passed = min(self.state[2], np.random.randint(1, 5)) + min(self.state[3], np.random.randint(1, 5))
            self.state[2] = max(0, self.state[2] - passed)
            self.state[3] = max(0, self.state[3] - passed)

        new_cars = np.random.randint(0, 3, size=4)
        self.state += new_cars
        reward = passed

        return self.state, reward

    def reset(self):
        self.state = np.random.randint(0, 10, size=4)
        self.current_light = 0
        return self.state


# Test the environment
env = TrafficEnvironment()
state = env.reset()

print("🚦 Initial Traffic Light Simulation 🚦")
print("State Format: [Cars North, Cars South, Cars East, Cars West]")
print("Actions: 0 = Keep Current Light, 1 = Switch Light")
print(f"Initial state: {state} (NS Green, EW Red)")

for i in range(10):
    action = np.random.choice([0, 1])
    next_state, reward = env.step(action)

    light_status = "NS Green, EW Red" if env.current_light == 0 else "EW Green, NS Red"
    print(f"Step {i + 1}:")
    print(f"  ➡ Action Taken: {action} ({'Switch' if action == 1 else 'Keep'})")
    print(f"  🚥 Light Status: {light_status}")
    print(f"  🏎 Cars State: {next_state}")
    print(f"  🎯 Reward (Cars Passed): {reward}")
    print("-" * 40)