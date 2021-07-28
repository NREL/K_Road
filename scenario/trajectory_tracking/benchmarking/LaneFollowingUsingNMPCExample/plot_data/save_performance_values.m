function save_performance_values(logs, ct_times, Xref, Yref, v_seq)
    
    for i = 1:size(logs, 1)
        for j = 1:size(logs, 2)
            if (j == 1)
                name = strcat("logs_perf_nl_", num2str(i));
            else
                name = strcat("logs_perf_linear_", num2str(i));
            end
            logsout_to_json(logs(i, j), strcat(name, ".json"));
        end  
    end
 
    csvwrite('positions.csv',[Xref; Yref]');
    csvwrite('logs_perf_ct_time.csv',ct_times);
    csvwrite('set_speed.csv', v_seq);

end