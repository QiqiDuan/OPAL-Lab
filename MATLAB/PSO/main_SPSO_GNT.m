% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Numerical Experiments of SPSO_GNT for Single-Objective 
%   Real-Parameter Continuous Function Optimization.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

close all;
clc;

%% set experimental parameters

% set a random seed for initializing the population for plotting the
%   convergence curve with a controllable starting point
RAND_SEED_FOR_INI_POP = 20170504; 

% set whether output the sequence of function evaluations or not,
%   should be set according to specific requirements
IS_OUTPUT_SEQ_FUN_EVAL = true;

% save all the final analysis results into a special folder which has a
%   same name with the algorithm name
ALGO_NAME = 'SPSO_GNT';
if ~exist(ALGO_NAME, 'dir')
    mkdir(ALGO_NAME);
end

FHD                = str2func('benchmark_fun');  % function handler for benchmark_fun.m
TOTAL_NUM_FUNS     = 6;                       % total number of test functions
%   index for all the test functions, range from 1 to TOTAL_NUM_FUNS
ind_fun            = 0;
TOTAL_NUM_TRIALS   = 50;                      % the total number of trials
%   index for all the trials, range from 1 to TOTAL_NUM_TRIALS
ind_trial          = 0;
FUN_DIM            = 100;                     % function dimension
MAX_FUN_EVAL       = 1e4 * FUN_DIM;           % maximum of function evaluations
POP_SIZE           = 100;                     % population size
MAX_ITER           = MAX_FUN_EVAL / POP_SIZE; % maximum of iterations (generations)

if MAX_ITER ~= fix(MAX_ITER)
    error(['\n\n\nERROR ---> main_SPSO_GNT.m -> experimental parameter ' ...
        '<MAX_ITER> is not an integer.']);
end

% for all the benchmark functions, each dimension has the same search bound, 
%   ranging from -100 to 100.
SEARCH_LOWER_BOUND = -100 * ones(POP_SIZE, FUN_DIM); % search lower bounds
SEARCH_UPPER_BOUND = +100 * ones(POP_SIZE, FUN_DIM); % search upper bounds

%% invoke PSO to optimize
for ind_fun = 1 : TOTAL_NUM_FUNS
    % initialize variables for performance statistics
    opt_pos = inf * ones(TOTAL_NUM_TRIALS, FUN_DIM); % optimal positions
    opt_val = inf * ones(TOTAL_NUM_TRIALS, 1);       % optimal values
    run_time = inf * ones(TOTAL_NUM_TRIALS, 1);      % run time
    if IS_OUTPUT_SEQ_FUN_EVAL % sequence of function evaluations
        seq_fun_eval = inf * ones(TOTAL_NUM_TRIALS, MAX_FUN_EVAL);
    else
        seq_fun_eval = inf * ones(TOTAL_NUM_TRIALS, 1);
    end
    
    % do trials for function optimization
    for ind_trial = 1 : TOTAL_NUM_TRIALS
        % give tips for long-run programs
        fprintf(sprintf('ind_fun = %02d && ind_trial = %02d ', ind_fun, ind_trial));
        % random seed for initializing the population
        INI_SEED = RAND_SEED_FOR_INI_POP + 1e4 * FUN_DIM + 1e2 * ind_fun + ind_trial;
        [opt_pos(ind_trial, :), opt_val(ind_trial, 1), ...
            seq_fun_eval(ind_trial, :), run_time(ind_trial, 1)] = ...
            SPSO_GNT(FHD, ind_fun, FUN_DIM, SEARCH_LOWER_BOUND, SEARCH_UPPER_BOUND, ...
            POP_SIZE, MAX_ITER, INI_SEED, IS_OUTPUT_SEQ_FUN_EVAL);
        fprintf(sprintf('elapsed time = %7.2f opt value = %7.5e\n', ...
            run_time(ind_trial, 1), opt_val(ind_trial, 1)));
    end
    fprintf('\n');
    
    % save all the final optimization results to the file system in the form of .mat
    %   opt ---> optima (optimal positions + optimal values)
    %   sfe --> sequence of function evaluations
    %   rts ---> run time summary
    save(sprintf('./%s/opt_Algo%s_Fun%02d_Dim%02d.mat', ...
        ALGO_NAME, ALGO_NAME, ind_fun, FUN_DIM), 'opt_pos', 'opt_val');
    save(sprintf('./%s/sfe_Algo%s_Fun%02d_Dim%02d.mat', ...
        ALGO_NAME, ALGO_NAME, ind_fun, FUN_DIM), 'seq_fun_eval');
    save(sprintf('./%s/rts_Algo%s_Fun%02d_Dim%02d.mat', ...
        ALGO_NAME, ALGO_NAME, ind_fun, FUN_DIM), 'run_time');
end
