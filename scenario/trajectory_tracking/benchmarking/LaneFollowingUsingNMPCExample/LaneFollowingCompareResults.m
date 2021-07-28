function LaneFollowingCompareResults(logsout1,logsout2, Xref, Yref)
% Plot and compare results for nonlinear and adaptive mpc. 

%% Get the data from simulation
% for nonlinear MPC
[e1_nmpc,e2_nmpc,delta_nmpc,accel_nmpc,vx_nmpc] = getData(logsout1);
% for adaptive MPC
[e1_ampc,e2_ampc,delta_ampc,accel_ampc,vx_ampc] = getData(logsout2);

%% Plot results. 
figure; % lateral results
% steering angle
subplot(5,1,1);
hold on;
grid on;
plot(delta_nmpc.Values.time,delta_nmpc.Values.Data);
plot(delta_ampc.Values.time,delta_ampc.Values.Data);
legend('Nonlinear MPC','Adaptive MPC');
title('Steering Angle (u2) vs Time');
xlabel('Time(s)');
ylabel('steering angle(radians)');
hold off;
% lateral deviation
subplot(5,1,2);
hold on;
grid on;
plot(e1_nmpc.Values.time,e1_nmpc.Values.Data);
plot(e1_ampc.Values.time,e1_ampc.Values.Data);
legend('Nonlinear MPC','Adaptive MPC');
title('Lateral Deviation (e1) vs Time');
xlabel('Time(s)');
ylabel('Lateral Deviation(m)');
hold off;
% relative yaw angle
subplot(5,1,3);
hold on;
grid on;
plot(e2_nmpc.Values.Time,e2_nmpc.Values.Data);
plot(e2_ampc.Values.Time,e2_ampc.Values.Data);
legend('Nonlinear MPC','Adaptive MPC');
title('Relative Yaw Angle (e2) vs Time');
xlabel('Time(s)');
ylabel('Relative Yaw Angle(radians)');
hold off;

nonlinear_info = logsout1.getElement("Info");
linear_info = logsout2.getElement("Info");

reference_x = logsout2.getElement("xref_out");
reference_y = logsout2.getElement("yref_out");

subplot(5,1,4);
hold on;
grid on;

%plot(logsout1{10}.Values.InertFrm.Cg.Disp.X.Data);
plot(nonlinear_info.Values.InertFrm.Cg.Disp.X.Time, nonlinear_info.Values.InertFrm.Cg.Disp.X.Data);
plot(linear_info.Values.InertFrm.Cg.Disp.X.Time, linear_info.Values.InertFrm.Cg.Disp.X.Data);
plot(reference_x.Values.Time, reference_x.Values.Data, '--k');
legend('Nonlinear MPC','Adaptive MPC');
title('X');
xlabel('Time(s)');
ylabel('X[m]');
hold off;

subplot(5,1,5);
hold on;
grid on;
plot(nonlinear_info.Values.InertFrm.Cg.Disp.Y.Time, nonlinear_info.Values.InertFrm.Cg.Disp.Y.Data);
plot(linear_info.Values.InertFrm.Cg.Disp.Y.Time, linear_info.Values.InertFrm.Cg.Disp.Y.Data);
plot(reference_y.Values.Time, reference_y.Values.Data, '--k');
legend('Nonlinear MPC','Adaptive MPC');
title('Y');
xlabel('Time(s)');
ylabel('Y[m]');
hold off;

figure; % longitudinal results
% acceleration
subplot(2,1,1);
hold on;
grid on;
plot(accel_nmpc.Values.time,accel_nmpc.Values.Data);
plot(accel_ampc.Values.time,accel_ampc.Values.Data);
legend('Nonlinear MPC','Adaptive MPC');
title('Acceleration (u1) vs Time');
xlabel('Time(s)');
ylabel('Acceleration(m/s^2)');
hold off;
% longitudinal velocity
subplot(2,1,2);
hold on;
grid on;
plot(vx_nmpc.Values.Time,vx_nmpc.Values.Data);
plot(vx_ampc.Values.Time,vx_ampc.Values.Data);
legend('Nonlinear MPC','Adaptive MPC');
title('Velocity (Vy) vs Time');
xlabel('Time(s)');
ylabel('Velocity(m/s)');
hold off;


nonlinear_info = logsout1.getElement("Info");
linear_info = logsout2.getElement("Info");

reference_l_x = logsout2.getElement("xref_out");
reference_l_y = logsout2.getElement("yref_out");
reference_nl_x = logsout1.getElement("xref_out");
reference_nl_y = logsout1.getElement("yref_out");

tnonlinear = nonlinear_info.Values.InertFrm.Cg.Disp.X.Time;
Xnonlinear = nonlinear_info.Values.InertFrm.Cg.Disp.X.Data;
Ynonlinear = nonlinear_info.Values.InertFrm.Cg.Disp.Y.Data;
t_nonlinear_sample = reference_nl_x.Values.Time;

tlinear = linear_info.Values.InertFrm.Cg.Disp.X.Time;
Xlinear = linear_info.Values.InertFrm.Cg.Disp.X.Data;
Ylinear = linear_info.Values.InertFrm.Cg.Disp.Y.Data;
t_linear_sample = reference_l_x.Values.Time;


interpolated_ref_x_nonlinear = interp1(t_nonlinear_sample, ... 
                                reference_nl_x.Values.Data,tnonlinear);
interpolated_ref_y_nonlinear = interp1(t_nonlinear_sample, ... 
                                reference_nl_y.Values.Data,tnonlinear);

interpolated_ref_x_linear = interp1(t_linear_sample, ... 
                                reference_l_x.Values.Data,tlinear);
interpolated_ref_y_linear = interp1(t_nonlinear_sample, ... 
                                reference_l_y.Values.Data,tlinear);
                            
                            
crosstrack_linear = logsout2.getElement("Crosstrack_Error").Values.Data;
crosstrack_nonlinear = logsout1.getElement("Crosstrack_Error").Values.Data;
                    
max_linear = max(crosstrack_linear)
max_nonlinear = max(crosstrack_nonlinear)
mean_linear = mean(crosstrack_linear)
mean_nonlinear = mean(crosstrack_nonlinear)


%% Local function: Get data from simulation
function [e1,e2,delta,accel,vx] = getData(logsout)
e1 = logsout.getElement('Lateral Deviation');    % lateral deviation
e2 = logsout.getElement('Relative Yaw Angle');   % relative yaw angle
delta = logsout.getElement('Steering');          % steering angle
accel = logsout.getElement('Acceleration');      % acceleration of ego car
vx = logsout.getElement('Longitudinal Velocity');% velocity of host car