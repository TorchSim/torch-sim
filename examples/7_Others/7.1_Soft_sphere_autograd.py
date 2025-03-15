"""Plot the soft sphere potential energy vs distance using plotly."""

# /// script
# dependencies = [
#     "plotly>=6",
#     "kaleido",
# ]
# ///

import numpy as np
import torch
from plotly.subplots import make_subplots

from torch_sim.models.soft_sphere import soft_sphere_pair, soft_sphere_pair_force


sigma = 1.0
epsilon = 1.0
alpha = 2

# Generate distance values from 0.1*sigma to 2*sigma
dr = np.linspace(0.1 * sigma, 2 * sigma, 1000)
dr_tensor = torch.sqrt(torch.tensor(dr))

# Calculate potential energy
# Make dr_tensor require gradients for autograd
dr_tensor.requires_grad_(True)  # noqa: FBT003

# Calculate potential energy with gradients enabled
energy = soft_sphere_pair(dr_tensor, sigma, epsilon, alpha)

# Calculate force as negative gradient of energy with respect to distance
force = torch.autograd.grad(energy, dr_tensor, grad_outputs=torch.ones_like(energy))[0]
explicit_force = soft_sphere_pair_force(dr_tensor, sigma, epsilon, alpha)

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add energy trace
fig.add_scatter(
    x=dr,
    y=energy.detach().numpy(),
    name="Energy",
    secondary_y=False,
)

# Add force traces
fig.add_scatter(
    x=dr,
    y=force.detach().numpy(),
    name="Force (Autograd)",
    secondary_y=True,
)

fig.add_scatter(
    x=dr,
    y=explicit_force.detach().numpy(),
    name="Force (Explicit)",
    line=dict(dash="dash"),
    secondary_y=True,
)

# Add figure titles and labels
fig.update_layout(
    title="Soft Sphere Potential",
    xaxis_title="Distance (r/σ)",
)

# Update y-axes labels
fig.update_yaxes(title_text="Energy (ε)", secondary_y=False)
fig.update_yaxes(title_text="Force (ε/σ)", secondary_y=True)

fig.write_image("soft_sphere_grad_vs_explicit.pdf")
