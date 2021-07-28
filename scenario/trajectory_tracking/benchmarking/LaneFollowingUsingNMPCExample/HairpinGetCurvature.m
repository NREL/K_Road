function [rho, t, v_seq, Duration, Xref, Yref, phi0, xref, yref] = HairpinGetCurvature(Ts)
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
    % Length parameters:
    length_arm_1 = 50.;
    radius_turn = 20.;
    % Target speeds:
    target_speed_1 = 10.;
    target_speed_turn = 5.;
    target_speed_2 = 10.;
    % First leg: 
    Yref = linspace(0, length_arm_1, 1000);
    Xref = repelem(0, size(Yref, 2));
    v = repelem(target_speed_1, size(Yref, 2));
    DX1 = gradient(Xref);
    DY1 = gradient(Yref);
    t = cumsum(sqrt(DY1.^2+DX1.^2))./v;
    t(1) = 0.0;
    % Semi-Circle Trajectory:
    starting_angle = pi;
    end_angle = 0.0;
    num_circle_segments = 1000;
    angles = linspace(starting_angle, end_angle, num_circle_segments);
    center_circle = [radius_turn, length_arm_1];
    x_ref_circ = radius_turn * cos(angles) + center_circle(1, 1);
    y_ref_circ = radius_turn * sin(angles) + center_circle(1, 2);
    Xref = [Xref, x_ref_circ];
    Yref = [Yref, y_ref_circ];
    vel = repelem(target_speed_turn, size(angles, 2));
    DX1 = gradient(x_ref_circ);
    DY1 = gradient(y_ref_circ);
    t_end = t(end);
    v = [v, vel];
    t = [t cumsum(sqrt(DY1.^2+DX1.^2))./vel + t_end];
    
    t_end = t(end);

    num_segments = 1000;
    last_arm = repelem(Xref(end), num_segments);
    last_arm_y = linspace(Yref(1, end)-1., 0.0, ...
                             num_segments);
     
    vel = repelem(target_speed_2, num_segments);
    
    vel(1:2) = [];
    last_arm(1:2) = [];
    last_arm_y(1:2) = [];
    
    Xref = [Xref, last_arm];
    Yref = [Yref, last_arm_y];
    v = [v, vel];
    
    DX1 = gradient(last_arm);
    DY1 = gradient(last_arm_y);
    
    t_3 = cumsum(sqrt(DY1.^2+DX1.^2))./vel + t_end;

    t = [t t_3];
   
    DX = gradient(Xref);
    DY = gradient(Yref);
    D2Y = gradient(DY);
    D2X = gradient(DX);

    curvature = (DX.* D2Y - DY.* D2X) ...
                            ./(DX.^2+DY.^2).^(3/2);  

    Duration = size(Xref, 2);
    
    rho.Xref = Xref;
    rho.Yref = Yref;
  
    rho.time = t;
    rho.signals.values = curvature';
    
    xref.time = t;
    xref.signals.values = Xref';
    
    v_seq.time = t;
    v_seq.signals.values = v';
    
    yref.time = t;
    yref.signals.values = Yref';
    
    phi0 = pi/2;
end
