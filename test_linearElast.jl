using Ferrite
# using FerriteViz
# import GLMakie

using LinearAlgebra

qr = QuadratureRule{2, RefCube}(2)
ip = Lagrange{2, RefCube, 1}()
# cv = CellVectorValues(qr, ip)
cv = CellValues(qr, ip)

E = 70e9 #units: N/m^2
ν = 0.3
dim = 2
λ = E*ν / ((1 + ν) * (1 - 2ν))
μ = E / (2(1 + ν))
δ(i,j) = i == j ? 1.0 : 0.0
f = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

Celas = SymmetricTensor{4, dim}(f)

grid = generate_grid(Quadrilateral, (20, 20), Vec((0.,0.)), Vec((1.,1.)))

dh = DofHandler(grid)
add!(dh, :u, 2)
close!(dh)

ch = ConstraintHandler(dh)
dbc1 = Dirichlet(:u, getfaceset(grid, "bottom"), (x,t)->[0., 0.0], [1,2])
add!(ch, dbc1)
dbc2 = Dirichlet(:u, getfaceset(grid, "top"), (x,t)->[0, -0.1], [1,2])
add!(ch, dbc2)
close!(ch)

function symmetrize_lower!(K)
    for i in eachindex(size(K,1))
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end


function assemble_cell!(fe, ke, cell, cv, Celas, ue)

    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(fe, 0.0)
    nu = getnbasefunctions(cv)

    for q_point in 1:getnquadpoints(cv)
        ϵ = function_symmetric_gradient(cv, q_point, ue)
        σ =  Celas ⊡ ϵ
        dΩ = getdetJdV(cv, q_point)
        for i in 1:nu
            δεi  = shape_symmetric_gradient(cv, q_point, i)
            fe[i] -= δεi ⊡ σ * dΩ
            for j in 1:nu
                δεj = shape_symmetric_gradient(cv, q_point, j)
                ke[i, j] += (δεi  ⊡ Celas ⊡ δεj) * dΩ
            end
        end
    end

    # symmetrize_lower!(ke)

end

function do_assemble!(f, K, cv, Celas, u)
    assembeler = start_assemble(K, f)
    nu = getnbasefunctions(cv)
    ke = zeros(nu,nu)
    fe = zeros(nu)

    for cell in CellIterator(dh)
        cell_dofs = celldofs(cell)
        assemble_cell!(fe, ke, cell, cv, Celas, u[cell_dofs])
        assemble!(assembeler, cell_dofs, ke, fe)
    end

    return K
end

K = create_sparsity_pattern(dh)


f = zeros(ndofs(dh))
u = zeros(ndofs(dh))

update!(ch, 0.0)


do_assemble!(f, K, cv, Celas, zeros(ndofs(dh)))

apply!(K, f, ch)
u = K\f

# u = cholesky(Symmetric(K))\f

# apply!(u, ch)

# rhs = get_rhs_data(ch, K)
# apply_rhs!(rhs, f, ch)

# apply!(K, ch)
# apply_zero!(K, f, ch)

# u = K\f

# apply!(u, ch)

# dof_f = ch.free_dofs
# dof_p = ch.prescribed_dofs

# u[dof_f] = K[dof_f, dof_f]\(-K[dof_f, dof_p]*u[dof_p])

plotter = MakiePlotter(dh, u)
ferriteviewer(plotter)