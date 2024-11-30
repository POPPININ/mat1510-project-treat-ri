import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_ri_trs_data(num_agents=5, num_samples=20, timesteps=100, dt=0.1):
    """
    Generate data for a multi-agent system that is both rotationally invariant (RI)
    and time-reversal symmetric (TRS).

    Args:
        num_agents (int): Number of agents in the system.
        num_samples (int): Number of trajectory samples.
        timesteps (int): Number of timesteps per trajectory.
        dt (float): Time step size.

    Returns:
        torch.Tensor: Tensor of shape (num_samples, timesteps, num_agents, 4) containing [position, velocity].
    """
    positions = []
    velocities = []

    for _ in range(num_samples):
        # Initialize random positions and velocities for agents
        theta = np.random.uniform(0, 2 * np.pi, size=(num_agents,))
        pos = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # Shape: (num_agents, 2)
        vel = np.zeros_like(pos)
        vel[:, 0] = -pos[:, 1]  # v_x = -y
        vel[:, 1] = pos[:, 0]   # v_y = x

        traj_pos = []
        traj_vel = []

        for t in range(timesteps):
            traj_pos.append(pos)
            traj_vel.append(vel)

            # Update position and velocity
            pos = pos + vel * dt  # Update position
            vel[:, 0] = -pos[:, 1]  # Recompute velocity
            vel[:, 1] = pos[:, 0]

        positions.append(np.stack(traj_pos, axis=0))  # Shape: (timesteps, num_agents, 2)
        velocities.append(np.stack(traj_vel, axis=0))  # Shape: (timesteps, num_agents, 2)

    positions = np.array(positions)  # Shape: (num_samples, timesteps, num_agents, 2)
    velocities = np.array(velocities)  # Shape: (num_samples, timesteps, num_agents, 2)

    # Combine positions and velocities
    data = np.concatenate([positions, velocities], axis=-1)  # Shape: (num_samples, timesteps, num_agents, 4)
    return torch.tensor(data, dtype=torch.float32)

def plot_trajectories(data, num_samples=5):
    """
    Plot the trajectories of the first few agents in the data.

    Args:
        data (torch.Tensor): Tensor of shape (num_samples, timesteps, num_agents, 4).
        num_samples (int): Number of samples to plot.
    """
    data = data.numpy()
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5), sharex=True, sharey=True)

    for i in range(num_samples):
        ax = axs[i]
        sample = data[i]  # Shape: (timesteps, num_agents, 4)

        for agent in range(sample.shape[1]):  # Iterate over agents
            positions = sample[:, agent, :2]  # Get positions (x, y)
            ax.plot(positions[:, 0], positions[:, 1], label=f"Agent {agent+1}")

        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate data
    data = generate_ri_trs_data(num_agents=5, num_samples=2000, timesteps=20000, dt=1e-2)
    print(f"Generated data shape: {data.shape}")
    print(f"Sample data (first trajectory, first agent, first 5 timesteps):\n{data[0, :5, 0]}")

    # Save data
    torch.save(data, "multi_agent_ri_trs_data.pt")
    print("Data saved to 'multi_agent_ri_trs_data.pt'")

    # Plot trajectories
    plot_trajectories(data)
