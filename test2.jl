using Ferrite, NearestNeighbors, LinearAlgebra, SparseArrays

# Generate two grids
grid1 = generate_grid(Quadrilateral, (10, 10), Vec((0., 0.)), Vec((1.0, 1.0)))
grid2 = generate_grid(Quadrilateral, (10, 10), Vec((0., 1.01)), Vec((1.0, 2.01)))

# Merge the grids
grid = Grid([grid1.cells; grid2.cells], [grid1.nodes; grid2.nodes])

ip = Lagrange{2, RefCube, 2}()
qr = QuadratureRule{2, RefCube}(2)

dh = DofHandler(grid)
add!(dh, :u, 2, ip)
close!(dh)

# Boundary facets for both grids
boundaryFacets1 = collect(union(grid1.facetsets["left"], grid1.facetsets["bottom"], grid1.facetsets["right"], grid1.facetsets["top"]))
boundaryFacets2 = collect(union(grid2.facetsets["left"], grid2.facetsets["bottom"], grid2.facetsets["right"], grid2.facetsets["top"]))

# Function to get the center of a facet
function get_centerFacet(grid, fI::FacetIndex)
    cell = fI[1]
    face = fI[2]
    cell_coords = getcoordinates(grid, cell)
    if face != length(cell_coords)
        vec = (cell_coords[face] + cell_coords[face+1])/2
        return [vec[1], vec[2]]
    else
        vec = (cell_coords[face] + cell_coords[1])/2
        return [vec[1], vec[2]]
    end
end

# Initial positions of the facet centers
centerFacet1 = [get_centerFacet(grid1, fI)[j] for j in 1:2, fI in boundaryFacets1]
centerFacet2 = [get_centerFacet(grid2, fI)[j] for j in 1:2, fI in boundaryFacets2]

# Initialize KD trees for contact detection
kdtree1 = KDTree(centerFacet1)
kdtree2 = KDTree(centerFacet2)

# Define parameters for the simulation
total_time = 1.0
dt = 0.01
num_steps = Int(total_time / dt)
CO_TOL = 1e-1
k_p = 1e5  # Penalty stiffness coefficient

# Initialize velocities and accelerations
velocities = zeros(2, length(grid.nodes))
accelerations = zeros(2, length(grid.nodes))

# Mass properties
mass = 1.0
mass_matrix = mass * I

# External forces (e.g., gravity)
external_forces = zeros(2, length(grid.nodes))

# Simulation loop
for step in 1:num_steps
    # Update positions using explicit Euler method
    for i in 1:length(grid.nodes)
        grid.nodes[i] .+= dt * velocities[:, i] + 0.5 * dt^2 * accelerations[:, i]
    end

    # Update velocities
    for i in 1:length(grid.nodes)
        velocities[:, i] .+= dt * accelerations[:, i]
    end

    # Reset accelerations to account for external forces
    accelerations .= mass_matrix \ external_forces

    # Detect contacts and apply penalty forces
    centerFacet1 = [get_centerFacet(grid1, fI)[j] for j in 1:2, fI in boundaryFacets1]
    centerFacet2 = [get_centerFacet(grid2, fI)[j] for j in 1:2, fI in boundaryFacets2]

    kdtree1 = KDTree(centerFacet1)
    kdtree2 = KDTree(centerFacet2)

    kdInfo_grid1 = [nn(kdtree2, point) for point in eachcol(centerFacet1)]
    kdInfo_grid2 = [nn(kdtree1, point) for point in eachcol(centerFacet2)]

    # Apply penalty forces
    for (i, kdinfo) in enumerate(kdInfo_grid1)
        if kdinfo[2] < CO_TOL
            contact_node = boundaryFacets1[i][1]
            penalty_force = k_p * (CO_TOL - kdinfo[2]) * [0, -1]  # Simplified normal force in y-direction
            accelerations[:, contact_node] .+= mass_matrix \ penalty_force
        end
    end

    for (i, kdinfo) in enumerate(kdInfo_grid2)
        if kdinfo[2] < CO_TOL
            contact_node = boundaryFacets2[i][1]
            penalty_force = k_p * (CO_TOL - kdinfo[2]) * [0, 1]  # Simplified normal force in y-direction
            accelerations[:, contact_node] .+= mass_matrix \ penalty_force
        end
    end

    # Optionally, apply damping to simulate energy loss
    velocities .*= 0.98
end

# Output final positions of the grids (for visualization or further analysis)
final_positions = [grid.nodes[i] for i in 1:length(grid.nodes)]
