% Projector-Camera Stereo calibration parameters:

% Intrinsic parameters of camera:
fc_left = [ 3572.699771 3554.303159 ]; % Focal Length
cc_left = [ 2461.074657 1264.313162 ]; % Principal point
alpha_c_left = [ 0.000000 ]; % Skew
kc_left = [ 0.003268 0.204619 -0.001989 0.019872 0.000000 ]; % Distortion

% Intrinsic parameters of projector:
fc_right = [ 1332.401845 2364.354831 ]; % Focal Length
cc_right = [ 554.247635 762.994844 ]; % Principal point
alpha_c_right = [ 0.000000 ]; % Skew
kc_right = [ -0.003217 0.297122 -0.008014 0.016674 0.000000 ]; % Distortion

% Extrinsic parameters (position of projector wrt camera):
om = [ -0.127498 0.331783 -0.052835 ]; % Rotation vector
T = [ -139.382159 -150.501649 195.217730 ]; % Translation vector
