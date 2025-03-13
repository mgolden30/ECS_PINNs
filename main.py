import numpy as np
import torch
import time

import torch.optim as optim
from torch import nn
from lib.StreamfunctionNetwork import StreamfunctionNetwork
from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set PyTorch seed for reproducibility
seed_value = 1
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

learning_rate = 1e-2
epochs = 1024

nu = 1.0/40 #fluid viscosity
nx = 64 #points in space
nt = 64 #points in time

#must start with 6 and end with 1
widths = [6, 64, 64, 64, 1]

def generate_uniform_grid(nx, nt):
    #Both space and time are nondimensionalized to [0,2pi]
    space = torch.linspace( 0, nx-1, nx, requires_grad=False ) / nx * 2 * np.pi
    time_ = torch.linspace( 0, nt-1, nt, requires_grad=False ) / nt * 2 * np.pi
    
    [x,y,t] = torch.meshgrid( space, space, time_ )

    x  = torch.reshape( x, [-1,1])
    y  = torch.reshape( y, [-1,1])
    t  = torch.reshape( t, [-1,1])
    xs = torch.cat((x,y,t), axis=1)

    k = torch.linspace( 0, nx-1, nx, requires_grad=False )
    k[k>nx//2] = k[k>nx//2] - nx
  
    kt = torch.linspace( 0, nt-1, nt, requires_grad=False )
    kt[kt>nt//2] = kt[kt>nt//2] - nt

    kx = torch.reshape(k,  [nx, 1, 1])
    ky = torch.reshape(k,  [1, nx, 1])
    kt = torch.reshape(kt, [1, 1, nt])

    #2/3rds dealiasing mask
    mask = (abs(kx) < nx/3) * (abs(ky) < nx/3) * (abs(kt) < nt/3)

    return xs, kx, ky, kt, mask



xs, kx, ky, kt, mask = generate_uniform_grid(nx,nt)

# -4cos(4y)
forcing = -4*torch.cos(4*xs[:,1])
forcing = torch.reshape( forcing, [nx, nx, nt] ).to(device)

t0 = 6.0
a0 = 0.0
aux = torch.tensor( [t0, a0] ).to(device)

network = StreamfunctionNetwork(widths, t0, a0)

psi = network.forward(xs)


network = network.to(device)
xs = xs.to(device)
kx = kx.to(device)
ky = ky.to(device)
kt = kt.to(device)
mask = mask.to(device)

def eval_loss( xs, kx, ky, kt, mask ):
    #Evaluate the streamfunction
    psi = network(xs)

    #Reshape
    psi = torch.reshape( psi, [nx, nx, nt] )

    #Take the Fourier transform in all spatiotemporal dimensions
    psi = torch.fft.fftn(psi)

    #Apply a 2/3rds dealiasing
    psi = psi * mask

    #Compute vorticity, gradients of vorticity, flow, ...
    w   = psi * (kx*kx + ky*ky) 
    wt  =  torch.fft.ifftn(  w * 1j * kt )
    wx  =  torch.fft.ifftn(  w * 1j * kx )
    wy  =  torch.fft.ifftn(  w * 1j * ky )
    u   =  torch.fft.ifftn(  psi * 1j * ky ) + aux[1]
    v   =  torch.fft.ifftn( -psi * 1j * kx )
    lap =  torch.fft.ifftn( - w * (kx*kx + ky*ky) )

    advection = u*wx + v*wy

    advection = torch.fft.ifftn( mask * torch.fft.fftn(advection) )

    navier_stokes =  (2*torch.pi / aux[0]) * wt + advection - forcing - nu*lap

    navier_stokes = torch.real(navier_stokes)

    criterion = nn.MSELoss()
    loss = criterion( navier_stokes, torch.zeros_like(navier_stokes) )
    return loss, navier_stokes

# Use an optimizer (e.g., Adam) to update the model parameters
optimizer = optim.Adam( network.parameters(), lr=learning_rate)
loss_history = torch.zeros( (epochs) )
walltimes = torch.zeros( epochs )

for epoch in range(epochs):
    
    t0 = time.time()
    loss, _ = eval_loss(xs, kx, ky, kt, mask)
    loss_history[epoch] = loss.detach()

    # Backward pass and optimization step
    optimizer.zero_grad()  # clear previous gradients
    loss.backward()  # compute gradients
    optimizer.step() # update model parameters
    t1 = time.time()

    walltimes[epoch] = t1 - t0

    # Print the loss every few epochs
    if epoch % 16 == 0:
        #save_network_output( hydro_model, "network_output/torch_output_%d.mat" % (epoch), "network_output/hydro_model_%d.pth" % (epoch) )
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, mean walltime = {torch.mean(walltimes[0:epoch])} seconds")

torch.save( network, 'model.pth' )

with torch.no_grad():
    psi = network(xs)
    _, navier_stokes = eval_loss(xs, kx, ky, kt, mask)


psi = torch.reshape( psi, [nx, nx, nt] )
psi = torch.fft.fftn(psi)
psi = psi * mask
w   = torch.fft.ifftn( psi * (kx*kx + ky*ky) )

savemat( "learned_state.mat", {"w": w.cpu().numpy(), "ns": navier_stokes.cpu().numpy(), "loss": loss_history} )