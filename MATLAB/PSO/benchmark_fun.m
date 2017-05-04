function y = benchmark_fun(x, fun_ind)
% *********************************************************************** %
% Benchmark Functions for Population-based Stochastic Optimization Algorithms,
%   Especially Evolutionary Computation.
%
% || INPUT  || <---
%   x       <--- matrix (pop_size, fun_dim)
%   fun_ind <--- integer, starting with 1
% || OUTPUT || --->
%   y ---> vector (1, pop_size)
%
% List for Benchmark Functions:
% -----------------------------
%   1. sphere: unimodal function
%   2. rosenbrock:
%   3. ackley: multimodal function, shallow local optima
%   4. griewanks: multimodal function
%   5. rastrigin: multimodal function
%   6. schwefel: multimodal function, deep local optima 
%
% Reference:
% ----------
%   * Liang J J, Qin A K, Suganthan P N, et al. Comprehensive learning 
%       particle swarm optimizer for global optimization of multimodal functions[J]. 
%       IEEE transactions on evolutionary computation, 2006, 10(3): 281-295.
% *********************************************************************** %
    switch fun_ind
        case 1
            y = sphere(x);
        case 2
            y = rosenbrock(x);
        case 3
            y = ackley(x);
        case 4
            y = griewanks(x);
        case 5
            y = rastrigin(x);
        case 6
            y = schwefel(x);
        otherwise
            error(['\n\n\nERROR ---> benchmark_fun.m -> provide an ' ...
                'invalid value (%d) for input argument <fun_ind>.\n\n\n'], fun_ind);
    end
    
    y = y';
end

% *********************************************************************** %
function y = sphere(x)
    y = sum(x .^ 2, 2);
end

function y = rosenbrock(x)
    [~, fun_dim] = size(x);
    if fun_dim < 2
        error(['\n\n\nERROR ---> benchmark_fun.m -> rosenbrock(x) -> ' ...
            '<fun_dim> of input argument x = %d (should >= 2).\n\n\n'], fun_dim)
    end
    y = 100 * sum((x(:, 1 : (fun_dim - 1)) .^ 2 - x(:, 2 : fun_dim)) .^ 2, 2) ...
        + sum((x(:, 1 : (fun_dim - 1)) - 1) .^ 2, 2);
end

function y = ackley(x)
    [~, fun_dim] = size(x);
    y = -20 * exp(-0.2 * sqrt(sum(x .^ 2, 2) / fun_dim)) ...
        - exp(sum(cos(2 * pi * x), 2) / fun_dim) + 20 + exp(1);
end

function y = griewanks(x)
    [pop_size, fun_dim] = size(x);
    y = sum(x .^ 2, 2) / 4000 ...
        - prod(cos(x ./ sqrt(repmat(1 : fun_dim, pop_size, 1))), 2) + 1;
end

function y = rastrigin(x)
    y = sum(x .^ 2 - 10 * cos(2 * pi * x) + 10, 2);
end

function y = schwefel(x)
    [~, fun_dim] = size(x);
    y = 418.9829 * fun_dim - sum(x .* sin(sqrt(abs(x))), 2);
end
