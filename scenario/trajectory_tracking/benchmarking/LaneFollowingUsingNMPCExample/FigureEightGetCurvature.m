function [rho, t, Duration, v, Xref, Yref, phi0, xref, yref] = FigureEightGetCurvature(Vx, v0, Ts)
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
    A = 500.;
    B = 500.;
    length = 10000.; 
    %v0 = 17;
    segment_length = 0.05; %v0*Ts;
    
    num_segments = ceil(length / segment_length);
    segments = 0:num_segments;
    
    Xref = A * cos(2. * pi * segments / num_segments);
    Yref = B * 0.5 * sin(4. * pi * segments / num_segments);
 
    DX = gradient(Xref);
    DY = gradient(Yref);
    D2Y = gradient(DY);
    D2X = gradient(DX);
    
    curvature = (DX.* D2Y - DY.* D2X) ...
                            ./(DX.^2+DY.^2).^(3/2);
    
    Duration = size(Xref, 2);
    
   
    t = cumsum(sqrt(DX.^2 + DY.^2))/v0; %linspace(0, length / v0, Duration);
    t_1 = t(1);
    t = t - t_1;

    v_seq = repelem(Vx, size(Xref, 2));
    
    rho.Xref = Xref;
    rho.Yref = Yref;
    rho.time = t;
    rho.signals.values = curvature';
    
    yref.time = t;
    yref.signals.values = Yref';
    xref.time = t;
    xref.signals.values = Xref';
    
    v.time = t;
    v.signals.values = v_seq';
    phi0 = pi/2;
end
