function print_status(iteration,overall,msg,clear)
%PRINT_STATUS prints the status of an iterative process
%
% Prints the percentage of a loop completed, and the estimated
% remaining time (based on the time necessary for the last iteration).
%
% iteration is the number of the current iteration
% overall   is the number of all iterations
% msg       is any message to be printed
%
% Example -> for i = 1:N ; ... print_status(i,N); end


    if ~exist('clear','var')
        clear = 1;
    end
    
    if clear
        clc;
    end
    
    if ~exist('msg','var')
        msg = '';
    end
    process = iteration / overall;
    if iteration == 1
        tic;
    end
    if mod(iteration,2)
      iteration_time = toc;
      iteration_time = iteration_time / 2;
      tic;
    else
      iteration_time = toc;
    end
    remaining = iteration_time * (overall - iteration);
    [h m s]   = sec2hms(remaining);
    fprintf('%s\nCompleted %.2f%% \n\nRemaining time: %d hours %d minutes %.0f seconds\n',msg,process*100,h,m,s);
end
