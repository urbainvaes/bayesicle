module Ackley

import Random
import Statistics

shift = 2;
n = 2;
# exact = shift * ones(n);

function rastrigin(x)
    A = 10
    result = A*n
    rootpow = 1
    for i in 1:n
        z = rootpow^i * (x[i] - shift)
        result += z^2 - A*cos(2π*z)
    end
    return result
end

function ackley(x)
    A = 20 + exp(1)
    z = x .- shift
    result = 20 + exp(1) - 20 * exp(-.2*sqrt(1/n*sum(abs2, z))) - exp(1/n*sum(cos.(2π*z)))
end

radius = 3*sqrt(2)
function sphere_constraint(x)
    return x'x .- radius^2
end

function grad_sphere_constraint(x)
    return x
end

end
