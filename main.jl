using Ferrite
using Tensors
using NearestNeighbors
using FerriteViz
import GLMakie

grid1 = generate_grid(Quadrilateral, (10, 10), Vec((0.,0.)), Vec((3.,3.)))
grid2 = generate_grid(Quadrilateral, (5,5), Vec((1.,4.)), Vec((2.,6.)))

nNodes1 = getnnodes(grid1)


for cellid in eachindex(grid2.cells)
    nodes = nNodes1.+grid2.cells[cellid].nodes
    grid2.cells[cellid] = Quadrilateral(nodes)
end

grid = Grid([grid1.cells; grid2.cells], [grid1.nodes; grid2.nodes])
# FerriteViz.wireframe(grid)

surf1 = grid1.facetsets["top"]

surf2 = grid2.facetsets["bottom"]

addnodeset!(grid, "top", x->x[2]≈3.0)
addnodeset!(grid, "bottom", x->x[2]≈4.0)

nodes1 = getnodes(grid, "top")
nodes2 = getnodes(grid, "bottom")

points1 = zeros(2, length(nodes1))
points2 = zeros(2, length(nodes2))

for idnode in eachindex(nodes1)
    points1[1, idnode] = nodes1[idnode].x[1]
    points1[2, idnode] = nodes1[idnode].x[2]
end

for idnode in eachindex(nodes2)
    points2[1, idnode] = nodes2[idnode].x[1]
    points2[2, idnode] = nodes2[idnode].x[2]
end

tree = KDTree(points1)
idxs, dists = nn(tree, points2)

# gᵀ*KN*g
KN = 1

# linear elasticity
# Material parameters
E = 200 #units: KN/m^2
ν = 0.3
dim = 2
λ = E*ν / ((1 + ν) * (1 - 2ν))
μ = E / (2(1 + ν))
δ(i,j) = i == j ? 1.0 : 0.0
f = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

Celas = SymmetricTensor{4, dim}(f)

qr = QuadratureRule{2, RefCube}(2)
ip = Lagrange{2, RefCube, 1}()
cv = CellValues(qr, ip^2)

dh = DofHandler(grid)
add!(dh, :u, 2, ip)
close!(dh)

addfacetset!(grid, "surf1", x->x[2]≈0.0)
addfacetset!(grid, "surf2", x->x[2]≈6.0)
addfacetset!(grid, "surf3", x->x[2]≈4.0)

ch = ConstraintHandler(dh)
dbc = Dirichlet(:u, getfacetset(grid, "surf1"), (x,t)->[0.0, 0.0], [1,2])
add!(ch, dbc)
dbc = Dirichlet(:u, getfacetset(grid, "surf2"), (x,t)->[0.0, -1.5], [1,2])
add!(ch, dbc)
dbc = Dirichlet(:u, getfacetset(grid, "surf3"), (x,t)->[0.0, 0.0], [1,2])
close!(ch)


function symmetrize_lower!(K)
    for i in eachindex(size(K,1))
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end


function assemble_cell!(fe, ke, cell, cv, Celas, ue)
    nu = getnbasefunctions(cv)
    reinit!(cv, cell)

    for q_point in 1:getnquadpoints(cv)
        ϵ = function_symmetric_gradient(cv, q_point, ue)
        σ =  Celas ⊡ ϵ
        dΩ = getdetJdV(cv, q_point)
        for i in 1:nu
            δϵi = shape_symmetric_gradient(cv, q_point, i)
            fe[i] -= ϵ ⊡ σ * dΩ
            for j in 1:nu
                δϵj = shape_symmetric_gradient(cv, q_point, j)
                ke[i, j] += (δϵi ⊡ Celas ⊡ δϵj) * dΩ
            end
        end
    end

    # symmetrize_lower!(ke)

end

function do_assemble!(f, K, dh, cv, Celas, u)
    assembeler = start_assemble(K, f)
    nu = ndofs_per_cell(dh)
    ke = zeros(nu,nu)
    fe = zeros(nu)

    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        cell_dofs = celldofs(cell)
        assemble_cell!(fe, ke, cell, cv, Celas, u[cell_dofs])
        assemble!(assembeler, cell_dofs, ke, fe)
    end

    return K
end

# for contact-penalty
addnodeset!(grid, "surf1", x->x[2]≈3.0)
addnodeset!(grid, "surf2", x->x[2]≈4.0)

nodes1 = getnodes(grid, "surf1")
nodes2 = getnodes(grid, "surf2")

points1 = zeros(2, length(nodes1))
points2 = zeros(2, length(nodes2))

for idnode in eachindex(nodes1)
    points1[1, idnode] = nodes1[idnode].x[1]
    points1[2, idnode] = nodes1[idnode].x[2]
end

for idnode in eachindex(nodes2)
    points2[1, idnode] = nodes2[idnode].x[1]
    points2[2, idnode] = nodes2[idnode].x[2]
end

tree1 = KDTree(points1)
idxs2, dists2 = nn(tree, points2)

tree2 = KDTree(points2)
idxs1, dists1 = nn(tree, points1)

surf1 = collect(grid.nodesets["surf1"])
surf2 = collect(grid.nodesets["surf2"])

K = create_sparsity_pattern(dh)

f = zeros(ndofs(dh))
update!(ch, 0.0)

# apply!(u, ch)

do_assemble!(f, K, dh, cv, Celas, zeros(ndofs(dh)))



for i in eachindex(surf2)
    @show id1 = surf1[idxs[i]]
    @show id2 = surf2[i]
    dof = [2*id1-1, 2*id1, 2*id2-1, 2*id2]
    K[dof, dof] += [0 0 0 0;0 KN 0 -KN; 0 0 0 0;0 -KN 0 KN]
end


# dof_f = ch.free_dofs
# dof_p = ch.prescribed_dofs

# u[dof_f] = K[dof_f, dof_f]\(f[dof_f] - K[dof_f, dof_p]*u[dof_p])
# 

apply!(K, f, ch)
u = K\f

# fig = GLMakie.Figure()
# axs = [GLMakie.Axis(fig[1,1]), GLMakie.Axis(fig[1,2])]
# plotter = MakiePlotter(dh, u)
# wireframe!(plotter)
# p1 = solutionplot!(axs[1], plotter, deformation_field=:u)
# fig[1,2] = GLMakie.Colorbar(fig[1,1], p1)
# fig
# ferriteviewer(plotter)

addnodeset!(grid, "boundary", x->x[2]≈3.0 || x[2]≈4.0)
addfacetset!(grid, "boundary", x->x[2]≈3.0 || x[2]≈4.0)

fv = FacetValues(QuadratureRule{1, RefCube}(2), ip^2)

boundaryNormals = Dict{Int, Vector{Float64}}()

contactB = getfacetset(grid, "boundary")

nodes_face = Dict(1=>(1,2), 2=>(2,3), 3=>(3,4), 4=>(4,1))

for (cell, face) in contactB
    n1_cell, n2_cell = nodes_face[face]
    n1 = grid.cells[cell].nodes[n1_cell]
    n2 = grid.cells[cell].nodes[n2_cell]
    coord1 = get_node_coordinate(grid, n1)
    coord2 = get_node_coordinate(grid, n2)
    temp_vec = coord2-coord1
    faceNormal = Vec((temp_vec[2], -temp_vec[1]))
    faceNormal /= norm(faceNormal)
    # if n2_cell+1>4
    #     coord_temp = get_node_coordinate(grid, n1_cell-1)
    #     temp_vec = 
    # else
    #     coord_temp = get_node_coordinate(grid, n2_cell+1)
    # end
    if haskey(boundaryNormals, n1)
        boundaryNormals[n1] = (boundaryNormals[n1]+ faceNormal)/2
    else
        push!(boundaryNormals, n1=>faceNormal)
    end
    if haskey(boundaryNormals, n2)
        boundaryNormals[n2] = (boundaryNormals[n2]+ faceNormal)/2
    else
        push!(boundaryNormals, n2=>faceNormal)
    end
end

function get_boundaryNodeNormal()
end

for facet in grid1.facetsets["top"]
end

function get_gapVector()
end

