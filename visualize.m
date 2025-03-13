clear;

load("learned_state_3.mat");

w = real(w);

for t = 1:size(w,3)
  tiledlayout(1,2);

  nexttile

  imagesc( w(:,:,t).' );
  axis square;
  colorbar();
  clim([-10 10]);
  colormap jet;
  
  nexttile
  semilogy(loss);
  ylabel("Navier-Stokes Loss");
  xlabel("epoch");
  axis square

  drawnow;
end
