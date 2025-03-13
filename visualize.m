clear;

%load("states/learned_state_3.mat");
load("learned_state.mat")

w = real(w);

for t = 1:size(w,3)
  tiledlayout(1,2);

  nexttile

  imagesc( w(:,:,t).' );
  axis square;
  colorbar();
  clim([-1 1]*10);
  colormap jet;
  
  nexttile
  semilogy(loss);
  ylabel("Navier-Stokes Loss");
  xlabel("epoch");
  axis square

  drawnow;
end
