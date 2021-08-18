module Ackley

import Random
import Statistics

shift = 2
n = 2
exact = shift * ones(n, 1);

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

function sphere_constraint(x)
    return sum(x'x) - 1
end

end
