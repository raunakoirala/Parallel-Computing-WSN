#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


#include "header_file.h"

NodeReport *node_reports;



/**
 * This is the main function that initializes the MPI environment, configures the grid of nodes for 
 * the electric vehicle charging simulation, and starts the base station or node processes based on the rank.
 */
int main(int argc, char **argv)
{
    int rank, size, nrows, ncols;
    int dims[NDIMS] = {0}, coord[NDIMS];
    int iterations = 3;            // Number of iterations to run the simulation
    MPI_Comm node_comm, node_grid_comm;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    int base = size - 1;  // The rank of the base station is the last rank

    // Create a new communicator where the base station belongs to a separate group
    MPI_Comm_split(MPI_COMM_WORLD, rank == base, 0, &node_comm);

    // If nrows and ncols are provided as command-line arguments, use them
    if (argc == 3) {
        nrows = atoi(argv[1]);
        ncols = atoi(argv[2]);

        // Ensure nrows * ncols equals to the number of node processes (excluding base station)
        if ((nrows * ncols) != size - 1) {
            if (rank == base)
                printf("ERROR: nrows*ncols)=%d *%d = %d != %d+1\n", nrows, ncols, nrows * ncols, size);
            MPI_Finalize();
            return 0;
        }
        dims[0] = nrows;
        dims[1] = ncols;
    } else {
        // Otherwise, let MPI decide the best grid dimensions
        MPI_Dims_create(size - 1, NDIMS, dims);
    }

    // If this is the base station process
    if (rank == base) {
        // Initialize the node reports
        node_reports = (NodeReport *)malloc(sizeof(NodeReport) * (size - 1));
        for (int i = 0; i < size - 1; i++) {
            node_reports[i].has_reported = 0;
        }

        // Start the base station function
        base_func(base, MPI_COMM_WORLD, &node_comm, dims, size, iterations);
    } else {
        // For all other processes (nodes)

        // Initialize the charging grid
        initialise_charging_grid(size - 1, rank, NDIMS, dims, node_comm, &node_grid_comm);

        // Start the simulation for the electric vehicle charging node
        simulate_ev_charging_node(rank, NDIMS, node_grid_comm);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}