using Integrals
using FastGaussQuadrature
using Cuba
using ProgressMeter
using Plots
using BSplines
using SparseArrays
using LinearAlgebra
using PrettyTables
using CSV, DataFrames

println("Number of threads: ", Threads.nthreads())

println("Defining constants...")
Z1 = 1.0
Z2 = 1.0
m = 1.0
c = 137.035999
R = 1.0
ξ_max = 30.0
ξ_min = 1.0
jz = 1 / 2

μ1 = jz - 1 / 2
μ2 = jz + 1 / 2;

println("Defining B-spline parameters and functions...")
k = 8
n_bsplines = 20
η_slope = 4e-2
save_sep = false

function indexToPair(index::Int, n::Int)
    """
    Convert a single index to a pair of indices.
    """
    i = div(index - 1, n) + 1
    j = mod(index - 1, n) + 1
    return i, j
end

println("Defining knot vector and GaussLegendre Nodes...")
ξ_knot = [ξ_min * (ξ_max / ξ_min)^((i) / (n_bsplines - k + 2)) for i in 0:(n_bsplines-k+2)]

η_knot = Float64.(vcat(-ones(1),
    [η_slope * (1 / η_slope)^(2 * (i - 1) / (n_bsplines - k - 1)) - 1 for i in 1:(n_bsplines-k)÷2],
    [1 - η_slope * (1 / η_slope)^(2 * (i - 1) / (n_bsplines - k - 1)) for i in (n_bsplines-k)÷2:-1:1],
    ones(1)))


bsplines_ξ = BSplineBasis(k, ξ_knot) # Warning: Don't use the last b-Spline

bsplines_η = BSplineBasis(k, η_knot)

x, w = gausslegendre(3 * k)

function int_redim_GL(f, a, b, i, j)
    t = ((b + a) / 2) .+ (b - a) * x / 2
    return (b - a) * sum(w .* f.(t, i, j)) / 2
end

println("Computing integrals for c11one...")
c11one_xi_fun_1(ξ, i, j) = (ξ^2 * m * c^2 - (Z1 + Z2) * ξ / R) * (ξ^2 - 1)^(jz - 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)
c11one_xi_fun_2(ξ, i, j) = (ξ^2 - 1)^(jz - 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)

c11one_eta_fun_1(η, i, j) = (1 - η^2)^(jz - 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)
c11one_eta_fun_2(η, i, j) = (-η^2 * m * c^2 + (Z1 - Z2) * η / R) * (1 - η^2)^(jz - 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)

int_c11one_xi_1 = zeros(n_bsplines, n_bsplines)
int_c11one_xi_2 = zeros(n_bsplines, n_bsplines)
int_c11one_eta_1 = zeros(n_bsplines, n_bsplines)
int_c11one_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines

    for l in 1:length(ξ_knot)-1
        int_c11one_xi_1[i, j] += int_redim_GL(c11one_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c11one_xi_2[i, j] += int_redim_GL(c11one_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c11one_eta_1[i, j] += int_redim_GL(c11one_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c11one_eta_2[i, j] += int_redim_GL(c11one_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c11one = zeros(n_bsplines^2, n_bsplines^2)
for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c11one[i, j] = 2 * pi * R^3 * (int_c11one_xi_1[i1, j1] * int_c11one_eta_1[i2, j2] + int_c11one_xi_2[i1, j1] * int_c11one_eta_2[i2, j2])
end

if save_sep==true
    CSV.write("c11one.csv", DataFrame(int_c11one, :auto), writeheader=false)
    println("int_c11one saved to c11one.csv")
end

println("Computing integrals for C22one...")
c22one_xi_fun_1(ξ, i, j) = (ξ^2 * m * c^2 - (Z1 + Z2) * ξ / R) * (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)
c22one_xi_fun_2(ξ, i, j) = (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)

c22one_eta_fun_1(η, i, j) = (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)
c22one_eta_fun_2(η, i, j) = (-η^2 * m * c^2 + (Z1 - Z2) * η / R) * (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)

int_c22one_xi_1 = zeros(n_bsplines, n_bsplines)
int_c22one_xi_2 = zeros(n_bsplines, n_bsplines)

int_c22one_eta_1 = zeros(n_bsplines, n_bsplines)
int_c22one_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_c22one_xi_1[i, j] += int_redim_GL(c22one_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c22one_xi_2[i, j] += int_redim_GL(c22one_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c22one_eta_1[i, j] += int_redim_GL(c22one_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c22one_eta_2[i, j] += int_redim_GL(c22one_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c22one = zeros(n_bsplines^2, n_bsplines^2)

for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c22one[i, j] = 2 * pi * R^3 * (int_c22one_xi_1[i1, j1] * int_c22one_eta_1[i2, j2] + int_c22one_xi_2[i1, j1] * int_c22one_eta_2[i2, j2])
end

if save_sep==true
    CSV.write("c22one.csv", DataFrame(int_c22one, :auto), writeheader=false)
    println("int_c22one saved to c22one.csv")
end

println("Computing integrals for C11two...")
c11two_xi_fun_1(ξ, i, j) = (ξ^2 * m * c^2 + (Z1 + Z2) * ξ / R) * (ξ^2 - 1)^(jz - 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)
c11two_xi_fun_2(ξ, i, j) = (ξ^2 - 1)^(jz - 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)

c11two_eta_fun_1(η, i, j) = (1 - η^2)^(jz - 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)
c11two_eta_fun_2(η, i, j) = (η^2 * m * c^2 + (Z1 - Z2) * η / R) * (1 - η^2)^(jz - 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)

int_c11two_xi_1 = zeros(n_bsplines, n_bsplines)
int_c11two_xi_2 = zeros(n_bsplines, n_bsplines)

int_c11two_eta_1 = zeros(n_bsplines, n_bsplines)
int_c11two_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_c11two_xi_1[i, j] += int_redim_GL(c11two_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c11two_xi_2[i, j] += int_redim_GL(c11two_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c11two_eta_1[i, j] += int_redim_GL(c11two_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c11two_eta_2[i, j] += int_redim_GL(c11two_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c11two = zeros(n_bsplines^2, n_bsplines^2)

for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c11two[i, j] = -2 * pi * R^3 * (int_c11two_xi_1[i1, j1] * int_c11two_eta_1[i2, j2] - int_c11two_xi_2[i1, j1] * int_c11two_eta_2[i2, j2])
end

if save_sep==true
    CSV.write("c11two.csv", DataFrame(int_c11two, :auto), writeheader=false)
    println("int_c11two saved to c11two.csv")
end

println("Computing integrals for C22two...")
c22two_xi_fun_1(ξ, i, j) = (ξ^2 * m * c^2 + (Z1 + Z2) * ξ / R) * (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)
c22two_xi_fun_2(ξ, i, j) = (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)

c22two_eta_fun_1(η, i, j) = (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)
c22two_eta_fun_2(η, i, j) = (η^2 * m * c^2 + (Z1 - Z2) * η / R) * (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)

int_c22two_xi_1 = zeros(n_bsplines, n_bsplines)
int_c22two_xi_2 = zeros(n_bsplines, n_bsplines)

int_c22two_eta_1 = zeros(n_bsplines, n_bsplines)
int_c22two_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_c22two_xi_1[i, j] += int_redim_GL(c22two_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c22two_xi_2[i, j] += int_redim_GL(c22two_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c22two_eta_1[i, j] += int_redim_GL(c22two_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c22two_eta_2[i, j] += int_redim_GL(c22two_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c22two = zeros(n_bsplines^2, n_bsplines^2)

for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c22two[i, j] = -2 * pi * R^3 * (int_c22two_xi_1[i1, j1] * int_c22two_eta_1[i2, j2] - int_c22two_xi_2[i1, j1] * int_c22two_eta_2[i2, j2])
end

if save_sep==true
    CSV.write("c22two.csv", DataFrame(int_c22two, :auto), writeheader=false)
    println("int_c22two saved to c22two.csv")
end

println("Computing integrals for C11three...")
c11three_xi_fun_1(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz + 1 / 2) * Function(BSpline(bsplines_ξ, j), Derivative(1), false)(ξ)
c11three_xi_fun_2(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz - 1 / 2) * ξ * BSpline(bsplines_ξ, j)(ξ)

c11three_eta_fun_1(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz - 1 / 2) * η * BSpline(bsplines_η, j)(η)
c11three_eta_fun_2(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz + 1 / 2) * Function(BSpline(bsplines_η, j), Derivative(1), false)(η)

int_c11three_xi_1 = zeros(n_bsplines, n_bsplines)
int_c11three_xi_2 = zeros(n_bsplines, n_bsplines)
int_c11three_eta_1 = zeros(n_bsplines, n_bsplines)
int_c11three_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_c11three_xi_1[i, j] += int_redim_GL(c11three_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c11three_xi_2[i, j] += int_redim_GL(c11three_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c11three_eta_1[i, j] += int_redim_GL(c11three_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c11three_eta_2[i, j] += int_redim_GL(c11three_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c11three = zeros(n_bsplines^2, n_bsplines^2)
for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c11three[i, j] = 2 * c * pi * R^2 * (int_c11three_xi_1[i1, j1] * int_c11three_eta_1[i2, j2] + int_c11three_xi_2[i1, j1] * int_c11three_eta_2[i2, j2])
end

if save_sep==true
    CSV.write("c11three.csv", DataFrame(int_c11three, :auto), writeheader=false)
    println("int_c11three saved to c11three.csv")
end

println("Computing integrals for C22three...")
c22three_xi_fun_1(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz + 3 / 2) * Function(BSpline(bsplines_ξ, j), Derivative(1), false)(ξ)
c22three_xi_fun_2(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz + 1 / 2) * ξ * BSpline(bsplines_ξ, j)(ξ)

c22three_eta_fun_1(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz + 1 / 2) * η * BSpline(bsplines_η, j)(η)
c22three_eta_fun_2(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz + 3 / 2) * Function(BSpline(bsplines_η, j), Derivative(1), false)(η)

int_c22three_xi_1 = zeros(n_bsplines, n_bsplines)
int_c22three_xi_2 = zeros(n_bsplines, n_bsplines)

int_c22three_eta_1 = zeros(n_bsplines, n_bsplines)
int_c22three_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_c22three_xi_1[i, j] += int_redim_GL(c22three_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c22three_xi_2[i, j] += int_redim_GL(c22three_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c22three_eta_1[i, j] += int_redim_GL(c22three_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c22three_eta_2[i, j] += int_redim_GL(c22three_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c22three = zeros(n_bsplines^2, n_bsplines^2)
for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c22three[i, j] = -2 * c * pi * R^2 * (int_c22three_xi_1[i1, j1] * int_c22three_eta_1[i2, j2] + int_c22three_xi_2[i1, j1] * int_c22three_eta_2[i2, j2])
end

if save_sep==true
CSV.write("c22three.csv", DataFrame(int_c22three, :auto), writeheader=false)
println("int_c22three saved to c22three.csv")
end

println("Computing integrals for C12three...")
c12three_xi_fun_1(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz - 1 / 2) * ξ^2 * BSpline(bsplines_ξ, j)(ξ)
c12three_xi_fun_2(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, j)(ξ)
c12three_xi_fun_3(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz - 1 / 2) * BSpline(bsplines_ξ, j)(ξ)
c12three_xi_fun_4(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz + 1 / 2) * ξ * Function(BSpline(bsplines_ξ, j), Derivative(1), false)(ξ)

c12three_eta_fun_1(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz - 1 / 2) * BSpline(bsplines_η, j)(η)
c12three_eta_fun_2(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, j)(η)
c12three_eta_fun_3(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz - 1 / 2) * η^2 * BSpline(bsplines_η, j)(η)
c12three_eta_fun_4(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz + 1 / 2) * η * Function(BSpline(bsplines_η, j), Derivative(1), false)(η)

int_c12three_xi_1 = zeros(n_bsplines, n_bsplines)
int_c12three_xi_2 = zeros(n_bsplines, n_bsplines)
int_c12three_xi_3 = zeros(n_bsplines, n_bsplines)
int_c12three_xi_4 = zeros(n_bsplines, n_bsplines)

int_c12three_eta_1 = zeros(n_bsplines, n_bsplines)
int_c12three_eta_2 = zeros(n_bsplines, n_bsplines)
int_c12three_eta_3 = zeros(n_bsplines, n_bsplines)
int_c12three_eta_4 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_c12three_xi_1[i, j] += int_redim_GL(c12three_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c12three_xi_2[i, j] += int_redim_GL(c12three_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c12three_xi_3[i, j] += int_redim_GL(c12three_xi_fun_3, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c12three_xi_4[i, j] += int_redim_GL(c12three_xi_fun_4, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c12three_eta_1[i, j] += int_redim_GL(c12three_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c12three_eta_2[i, j] += int_redim_GL(c12three_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
        int_c12three_eta_3[i, j] += int_redim_GL(c12three_eta_fun_3, η_knot[l], η_knot[l+1], i, j)
        int_c12three_eta_4[i, j] += int_redim_GL(c12three_eta_fun_4, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c12three = zeros(n_bsplines^2, n_bsplines^2)
for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c12three[i, j] = 2 * c * pi * R^2 * (μ2 * int_c12three_xi_1[i1, j1] * (int_c12three_eta_1[i2, j2] + int_c12three_eta_2[i2, j2]) + μ2 * int_c12three_eta_3[i2, j2] * (int_c12three_xi_2[i1, j1] - int_c12three_xi_3[i1, j1]) + int_c12three_xi_4[i1, j1] * int_c12three_eta_2[i2, j2] - int_c12three_xi_2[i1, j1] * int_c12three_eta_4[i2, j2])
end

if save_sep==true
CSV.write("c12three.csv", DataFrame(int_c12three, :auto), writeheader=false)
println("int_c12three saved to c12three.csv")
end

println("Computing integrals for C21three...")
c21three_xi_fun_1(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz + 1 / 2) * ξ * Function(BSpline(bsplines_ξ, j), Derivative(1), false)(ξ)
c21three_xi_fun_2(ξ, i, j) = BSpline(bsplines_ξ, i)(ξ) * (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, j)(ξ)

c21three_eta_fun_1(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, j)(η)
c21three_eta_fun_2(η, i, j) = BSpline(bsplines_η, i)(η) * (1 - η^2)^(jz + 1 / 2) * η * Function(BSpline(bsplines_η, j), Derivative(1), false)(η)

int_c21three_xi_1 = zeros(n_bsplines, n_bsplines)
int_c21three_xi_2 = zeros(n_bsplines, n_bsplines)
int_c21three_eta_1 = zeros(n_bsplines, n_bsplines)
int_c21three_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_c21three_xi_1[i, j] += int_redim_GL(c21three_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_c21three_xi_2[i, j] += int_redim_GL(c21three_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_c21three_eta_1[i, j] += int_redim_GL(c21three_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_c21three_eta_2[i, j] += int_redim_GL(c21three_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_c21three = zeros(n_bsplines^2, n_bsplines^2)
for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_c21three[i, j] = 2 * c * pi * R^2 * (int_c21three_xi_1[i1, j1] * int_c21three_eta_1[i2, j2] - int_c21three_xi_2[i1, j1] * int_c21three_eta_2[i2, j2])
end

if save_sep==true
CSV.write("c21three.csv", DataFrame(int_c21three, :auto), writeheader=false)
println("int_c21three saved to c21three.csv")
end

println("Computing integrals for S11...")
s11_xi_fun_1(ξ, i, j) = ξ^2 * (ξ^2 - 1)^(jz - 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)
s11_xi_fun_2(ξ, i, j) = (ξ^2 - 1)^(jz - 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)

s11_eta_fun_1(η, i, j) = (1 - η^2)^(jz - 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)
s11_eta_fun_2(η, i, j) = η^2 * (1 - η^2)^(jz - 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)

int_s11_xi_1 = zeros(n_bsplines, n_bsplines)
int_s11_xi_2 = zeros(n_bsplines, n_bsplines)

int_s11_eta_1 = zeros(n_bsplines, n_bsplines)
int_s11_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_s11_xi_1[i, j] += int_redim_GL(s11_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_s11_xi_2[i, j] += int_redim_GL(s11_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_s11_eta_1[i, j] += int_redim_GL(s11_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_s11_eta_2[i, j] += int_redim_GL(s11_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_s11 = zeros(n_bsplines^2, n_bsplines^2)
for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_s11[i, j] = 2 * pi * R^3 * (int_s11_xi_1[i1, j1] * int_s11_eta_1[i2, j2] - int_s11_xi_2[i1, j1] * int_s11_eta_2[i2, j2])
end

if save_sep==true
    CSV.write("s11one.csv", DataFrame(int_s11, :auto), writeheader=false)
    println("int_s11 saved to s11one.csv")
end

println("Computing integrals for S22...")
s22_xi_fun_1(ξ, i, j) = ξ^2 * (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)
s22_xi_fun_2(ξ, i, j) = (ξ^2 - 1)^(jz + 1 / 2) * BSpline(bsplines_ξ, i)(ξ) * BSpline(bsplines_ξ, j)(ξ)

s22_eta_fun_1(η, i, j) = (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)
s22_eta_fun_2(η, i, j) = η^2 * (1 - η^2)^(jz + 1 / 2) * BSpline(bsplines_η, i)(η) * BSpline(bsplines_η, j)(η)

int_s22_xi_1 = zeros(n_bsplines, n_bsplines)
int_s22_xi_2 = zeros(n_bsplines, n_bsplines)

int_s22_eta_1 = zeros(n_bsplines, n_bsplines)
int_s22_eta_2 = zeros(n_bsplines, n_bsplines)

@showprogress for i in 1:n_bsplines, j in 1:n_bsplines
    for l in 1:length(ξ_knot)-1
        int_s22_xi_1[i, j] += int_redim_GL(s22_xi_fun_1, ξ_knot[l], ξ_knot[l+1], i, j)
        int_s22_xi_2[i, j] += int_redim_GL(s22_xi_fun_2, ξ_knot[l], ξ_knot[l+1], i, j)
    end

    for l in 1:length(η_knot)-1
        int_s22_eta_1[i, j] += int_redim_GL(s22_eta_fun_1, η_knot[l], η_knot[l+1], i, j)
        int_s22_eta_2[i, j] += int_redim_GL(s22_eta_fun_2, η_knot[l], η_knot[l+1], i, j)
    end
end

int_s22 = zeros(n_bsplines^2, n_bsplines^2)
for i in 1:n_bsplines^2, j in 1:n_bsplines^2
    i1, i2 = indexToPair(i, n_bsplines)
    j1, j2 = indexToPair(j, n_bsplines)
    int_s22[i, j] = 2 * pi * R^3 * (int_s22_xi_1[i1, j1] * int_s22_eta_1[i2, j2] - int_s22_xi_2[i1, j1] * int_s22_eta_2[i2, j2])
end

if save_sep==true
    CSV.write("s22one.csv", DataFrame(int_s22, :auto), writeheader=false)
    println("int_s22 saved to s22one.csv")
end

println("Creating the Hamiltonian matrix...")
C_M = zeros(4 * n_bsplines^2, 4 * n_bsplines^2)

# Diagonal blocks
C_M[1:n_bsplines^2, 1:n_bsplines^2] = int_c11one
C_M[n_bsplines^2+1:2*n_bsplines^2, n_bsplines^2+1:2*n_bsplines^2] = int_c22one
C_M[2*n_bsplines^2+1:3*n_bsplines^2, 2*n_bsplines^2+1:3*n_bsplines^2] = int_c11two
C_M[3*n_bsplines^2+1:4*n_bsplines^2, 3*n_bsplines^2+1:4*n_bsplines^2] = int_c22two


# Off-diagonal blocks
C_M[1:n_bsplines^2, 2*n_bsplines^2+1:3*n_bsplines^2] = int_c11three
C_M[2*n_bsplines^2+1:3*n_bsplines^2, 1:n_bsplines^2] = int_c11three'
C_M[1:n_bsplines^2, 3*n_bsplines^2+1:4*n_bsplines^2] = int_c12three
C_M[3*n_bsplines^2+1:4*n_bsplines^2, 1:n_bsplines^2] = int_c12three'
C_M[n_bsplines^2+1:2*n_bsplines^2, 2*n_bsplines^2+1:3*n_bsplines^2] = int_c21three
C_M[2*n_bsplines^2+1:3*n_bsplines^2, n_bsplines^2+1:2*n_bsplines^2] = int_c21three'
C_M[n_bsplines^2+1:2*n_bsplines^2, 3*n_bsplines^2+1:4*n_bsplines^2] = int_c22three
C_M[3*n_bsplines^2+1:4*n_bsplines^2, n_bsplines^2+1:2*n_bsplines^2] = int_c22three'

println("Creating the overlap matrix...")
S_M = zeros(4 * n_bsplines^2, 4 * n_bsplines^2)

# Diagonal blocks
S_M[1:n_bsplines^2, 1:n_bsplines^2] = int_s11
S_M[n_bsplines^2+1:2*n_bsplines^2, n_bsplines^2+1:2*n_bsplines^2] = int_s22
S_M[2*n_bsplines^2+1:3*n_bsplines^2, 2*n_bsplines^2+1:3*n_bsplines^2] = int_s11
S_M[3*n_bsplines^2+1:4*n_bsplines^2, 3*n_bsplines^2+1:4*n_bsplines^2] = int_s22

println("Diagonalizing the problem...")

F = eigen(C_M, S_M)

E = F.values
E = E[isfinite.(E)]  # Remove infinite values

data = hcat(E, E .- m * c^2, abs.(E) .< m * c^2)

hl_1 = Highlighter(
    (data, i, j) -> (j == 1) && (abs(data[i, j]) <= m * c^2),
    crayon"blue bold"
);

hl_2 = Highlighter(
    (data, i, j) -> (j == 1) && (abs(data[i, j]) > m * c^2),
    crayon"red bold"
);

open("eigenvalues.txt", "w") do file
    pretty_table(file, data; header=["Eigenvalues", "Shifted Eigenvalues", "If Bound State"], highlighters=(hl_1, hl_2), tf=tf_unicode_rounded)
end

pretty_table(data; header=["Eigenvalues", "Shifted Eigenvalues", "If Bound State"], highlighters=(hl_1, hl_2), tf=tf_unicode_rounded)

println("Eigenvalues saved to eigenvalues.txt")