function [rho, t, Duration, v, Xref, Yref, phi0, xref, yref] = PulsedSpeedPathGetCurvature(Ts)
% [reference] Autonomous Vehicle Trajectory Tracking via 
% Model-free Deep Reinforcement Learning
% Tripp, C., Aguasvivas Manzano, S.,
% Zhang, X., Lunacek, M., Graf, P. IEEE Transactions on 
% Intelligent Transportation Systems. Submitted for review
% [author] Aguasvivas Manzano, S. 
% Modified from:
% https://www.mathworks.com/help/mpc/ug/
%    lane-following-using-nonlinear-model-predictive-control.html
% The plant model was modified to be the Vehicle 3DOF Driving Model
% from the Automated Driving Toolbox found here:
% https://www.mathworks.com/help/vdynblks/ref/vehiclebody3dof.html
% [title] Helper script declare the curvature and velocity profiles 
% of the path defined by the title of this function
    length = 3300.;
    starting_speed = 10.;
    pulse_speed = 40.;
    segment_length = 0.5;
    num_segments = ceil(length/ segment_length);
    
    Xref = [];
    Yref = [];
    vel = [];
    t = [];
    for i= 1:num_segments
        if i*segment_length < length / 3.
           target_speed = starting_speed; 
        elseif i*segment_length > length / 3. && i*segment_length < 2*length / 3.
            target_speed = pulse_speed;
        elseif i*segment_length > 2*length / 3
            target_speed = starting_speed;
        end
      
      Xref = [ Xref, i];
      Yref = [Yref, 0.0];
      vel = [vel, target_speed];
      t = [t (i-1)* segment_length / starting_speed];
    end
    
    % Desired curvature
    DX = gradient(Xref,0.1);
    DY = gradient(Yref,0.1);
    D2Y = gradient(DY,0.1);
    curvature = DX.*D2Y./(DX.^2+DY.^2).^(3/2);
    curvature(isnan(curvature)) = 0;
    
    Duration = size(Xref, 2);
    
    rho.Xref = Xref;
    rho.Yref = Yref;
    rho.time = t;
    rho.signals.values = curvature';
    
    yref.time = t;
    yref.signals.values = Yref';
    xref.time = t;
    xref.signals.values = Xref';
    
    v.time = t;
    v.signals.values = vel';
    phi0 = 0;
end
