simout = sim('Forward_Dynamics_Model2.slx','StartTime',...,
                         '0','StopTime', ...
                        '79.9','FixedStep','0.05');
logsout = simout.get('logsout');