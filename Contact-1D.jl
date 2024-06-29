using Ferrite
using SparseArrays

grid = Grid([Line((1,2)), Line((3,4))], [Node{1, Float64}(Vec((0.))), Node{1, Float64}(Vec((1.))),
                                         Node{1, Float64}(Vec((0.98))), Node{1, Float64}(Vec((2.)))])

dh = DofHandler(grid)

add!(dh, :u, 1)
close!(dh)

ip = Lagrange{1, RefCube, 1}()
qr = QuadratureRule{1, RefCube}(1) 

cv = CellValues(qr, ip, ip)

K = create_sparsity_pattern(dh)

ch = ConstraintHandler(dh);

# addnodeset!(grid, "2", (x)->x[1]>2.00)
addfacetset!(grid, "1", x -> x[1] ≈ 0.0)
addfacetset!(grid, "2", x -> x[1] ≈ 2.0)
∂Ω1 = getfacetset(grid, "1")
∂Ω2 = getfacetset(grid, "2")
dbc1 = Dirichlet(:u, ∂Ω1, x -> 0)
dbc2 = Dirichlet(:u, ∂Ω2, x -> -0.1)
add!(ch, dbc1)
add!(ch, dbc2)
close!(ch)

function assemble_element!(Ke::Matrix, fe::Vector, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cv)
        # Get the quadrature weight
        dΩ = getdetJdV(cv, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cv, q_point, i)
            ∇δu = shape_gradient(cv, q_point, i)
            # Add contribution to fe
            # fe[i] += δu * dΩ
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cv, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_global(cv::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Allocate global force vector f
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cels
    for cell in CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cv, cell)
        # Compute element contribution
        assemble_element!(Ke, fe, cv)
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end


function update_contactForce!(f, kn, u)
    # gn from node 2 to node 3
    gn = (grid.nodes[3].x[1] + u[3] - (grid.nodes[2].x[1] + u[2]))
    if gn<0
        f[2] += kn*gn
        f[3] -= kn*gn
    end
    return gn
end

update!(ch, 0.0)

K = create_sparsity_pattern(dh, ch)

K, f = assemble_global(cv, K, dh)
K_orig = deepcopy(K)
u = zeros(ndofs(dh))

kn = 1e10
K[2,2] += kn
K[2,3] -= kn
K[3,3] += kn
K[3,2] -= kn
K
# apply_zero!(K,f,ch)
# apply!(K, f, ch)
# K
# f

dof_f = ch.free_dofs
dof_p = ch.prescribed_dofs

apply!(u, ch)

gn = update_contactForce!(f, kn, u)

u[dof_f] .= K[dof_f, dof_f] \ (f[dof_f] - K[dof_f, dof_p]*u[dof_p])

K_orig*u

gn = update_contactForce!(f, kn, u)
u[dof_f] .= K[dof_f, dof_f] \ (f[dof_f] - K[dof_f, dof_p]*u[dof_p])
