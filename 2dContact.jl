using Ferrite, NearestNeighbors

grid1 = generate_grid(Quadrilateral, (10, 10), Vec((0., 0.)), Vec((1.0, 1.0)))
grid2 = generate_grid(Quadrilateral, (10, 10), Vec((0., 1.01)), Vec((1.0, 2.01)))

# merged grid
grid = Grid([grid1.cells; grid2.cells], [grid1.nodes; grid2.nodes])

ip = Lagrange{2, RefCube, 2}()
qr = QuadratureRule{2, RefCube}(2)

dh = DofHandler(grid)
add!(dh, :u, 2, ip)
close!(dh)

boundaryFacets1 = collect(union(grid1.facetsets["left"], grid1.facetsets["bottom"], grid1.facetsets["right"], grid1.facetsets["top"]))
boundaryFacets2 = collect(union(grid2.facetsets["left"], grid2.facetsets["bottom"], grid2.facetsets["right"], grid2.facetsets["top"]))

# reduced_boundFacet1 = grid1.facetsets["top"]
# reduced_boundFacet2 = grid2.facetsets["bottom"]

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

# centerFacet1_temp = get_centerFacet.(boundaryFacets1)
# centerFacet1 = [centerFacet1_temp[i][j] for j in 1:2, i in 1:length(boundaryFacets1)]

@run centerFacet1 = [get_centerFacet(grid1, fI)[j] for j in 1:2, fI in boundaryFacets1]

centerFacet2 = [get_centerFacet(grid2, fI)[j] for j in 1:2, fI in boundaryFacets2]

# ContactData = hcat(centerFacet1, centerFacet2)

kdtree1 = KDTree(centerFacet1)
kdtree2 = KDTree(centerFacet2)

kdInfo_grid1 = [nn(kdtree2, point) for point in eachcol(centerFacet1)]
kdInfo_grid2 = [nn(kdtree1, point) for point in eachcol(centerFacet2)]

CO_TOL = 1e-1

# To find the contact point for grid 1
findall(x->x[2]<CO_TOL, kdInfo_grid1)

# To find the contact point for grid 2
findall(x->x[2]<CO_TOL, kdInfo_grid2)
