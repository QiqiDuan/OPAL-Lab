function [opt_pos, opt_val, seq_fun_eval, run_time] = ...
    SPSO_GNT(fhd, ind_fun, fun_dim, slb, sub, pop_size, ...
    max_iter, ini_seed, is_output_seq_fun_eval)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Sequential (Standard) Particle Swarm Optimizer With a Global Neighbor Topology.
%
% || INPUT  || <---
%   fhd      <--- str2func, benchmark function handler
%   ind_fun  <--- integer, index for benchmark functions
%   fun_dim  <--- integer, benchmark function dimension
%   slb      <--- matrix(pop_size, fun_dim), search lower bound
%   sub      <--- matrix(pop_size, fun_dim), search upper bound
%   pop_size <--- integer, population (swarm) size
%   max_iter <--- integer, maximum of iterations (generations)
%   ini_seed <--- integer, random seed for initializing the population (particles)
%   is_output_seq_fun_eval <--- logical, whether ouput <seq_fun_eval>
%
% || OUTPUT || --->
%   opt_pos      ---> matrix(1, fun_dim), optimal function position (point/solution)
%   opt_val      ---> double, optimal function (cost/fitness) value
%   seq_fun_eval ---> matrix(1, pop_size * max_iter), sequence of function evaluations (FEs)
%   run_time     ---> double, run time of the program
%
% Reference:
% ----------
%   * Eberhart R, Kennedy J. A new optimizer using particle swarm theory.
%       Micro Machine and Human Science, 
%       Proceedings of the Sixth International Symposium on. IEEE, 1995: 39-43.
%   * Shi Y, Eberhart R. A modified particle swarm optimizer.
%       Evolutionary Computation Proceedings, 
%       IEEE World Congress on Computational Intelligence. 1998: 69-73.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
    % initialize experimental parameters
    run_time_start = tic;
    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', 'shuffle'));
    if is_output_seq_fun_eval
        seq_fun_eval = zeros(1, pop_size * max_iter);
    else
        seq_fun_eval = 'is_output_seq_fun_eval = false';
    end
    
    % initialize algorithmic parameters
    %   inertia_weights linearly decresing from 0.9 to 0.4
    w = linspace(0.9, 0.4, max_iter);
    c1 = 2; % cognition learning parameter
    c2 = 2; % social learning parameter
    
    vlb = 0.01 * slb; % velocity lower bounds
    vub = 0.01 * sub; % velocity upper bounds
    v = vlb + (vub - vlb) .* rand(pop_size, fun_dim); % velocities
    
    ini_rand = rand(RandStream('mt19937ar', 'Seed', ini_seed), pop_size, fun_dim);
    x = slb + (sub - slb) .* ini_rand; % positions
    fun_val = feval(fhd, x, ind_fun); % function evaluations
    if is_output_seq_fun_eval
        seq_fun_eval(1 : pop_size) = fun_val;
    end
    
    %   initialize the individually-best positions and function values
    pi = x;
    pi_fun_val = fun_val;
    
    %   initialize the globally-best positions and function values
    [pg_fun_val, ind_pg] = min(pi_fun_val);
    pg = repmat(x(ind_pg, :), pop_size, 1);
    
    % synchronous iterative update instead of asynchronous
    %   NOTE that the index of iterations starts from 2 rather than 1
    for ind_iter = 2 : max_iter
       % update the velocities
       v = w(ind_iter) .* v ...
           + c1 .* rand(pop_size, fun_dim) .* (pi - x) ...
           + c2 .* rand(pop_size, fun_dim) .* (pg - x);
       
       % limit the lower and upper bounds
       v = (v < vlb) .* vlb + (v >= vlb) .* v; 
       v = (v > vub) .* vub + (v <= vub) .* v;
       
       % update the positions
       x = x + v;
       
       % limit the search lower and upper bounds (via re-initialization)
       x_rand = slb + (sub - slb) .* rand(pop_size, fun_dim);
       x = (x < slb) .* x_rand + (x >= slb) .* x;
       x = (x > sub) .* x_rand + (x <= sub) .* x;
       
       % evaluate function values
       fun_val = feval(fhd, x, ind_fun);
       if is_output_seq_fun_eval
           seq_fun_eval(((ind_iter -1) * pop_size + 1) : (ind_iter * pop_size)) = fun_val;
       end
       
       % update the individually-best positions and function values
       ind_pi = fun_val < pi_fun_val;
       pi(ind_pi, :) = x(ind_pi, :);
       pi_fun_val(ind_pi) = fun_val(ind_pi);
       
       % update the globally-best positions and function values
       if min(fun_val) < pg_fun_val
           [pg_fun_val, ind_pg] = min(fun_val);
           pg = repmat(x(ind_pg, :), pop_size, 1);
       end
    end
    
    % output the final optimization results
    opt_pos = pg(1, :);
    opt_val = pg_fun_val;
    run_time = toc(run_time_start);
end
