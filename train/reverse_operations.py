# train/reverse_operations.py

def reverse_op(z):
    """
    Reverse operation: flip the velocity component.
    Args:
        z: Trajectory tensor (T, batch_size, num_agents, dim)
    Returns:
        Reverse trajectory tensor (T, batch_size, num_agents, dim)
    """
    z_reversed = z.clone()
    z_reversed[..., 2:] *= -1  # Assume velocity starts at dim 2
    return z_reversed
