#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>
#include <string.h>


#include "header_file.h"


/**
 * Manages communication between nodes to check and share their port statuses.
 * The function determines when to communicate with its neighboring nodes based on the number of busy ports.
 * When a certain threshold of busy ports is reached, it will gather information from neighboring nodes.
 * If all nodes in the vicinity, including the current node, are near their capacity, it alerts a base station.
 *
 * @param k                    The number of ports on the node.
 * @param port_status          An array of size 'k', representing the status of each port (1 for busy, 0 for free).
 * @param rank                 The rank of the current node in the MPI environment.
 * @param shared_comm          The MPI communicator associated with a grid of nodes.
 * @param shared_ndims         The number of dimensions in the grid topology of nodes.
 * @param global_terminate_flag A pointer to a flag that indicates if the node should terminate its operation.
 */
void communicate_nodes(int k, int* port_status, int rank, MPI_Comm shared_comm, int shared_ndims, int* global_terminate_flag) {

    // Cartesian coordinates of the current node in the MPI grid
    int coord[shared_ndims];
    MPI_Cart_coords(shared_comm, rank, shared_ndims, coord);

    // Constants to determine how often to check and when to trigger communication
    int check_period = 5;  // Check every 5 seconds
    int threshold = k - 2;  // Trigger communication if only 2 or fewer ports are free

    int message_count = 0;  // Count of messages sent

    // Variables for graceful termination
    int terminate = 0;
    MPI_Request termination_request;
    int terminate_received = 0;

    // Infinite loop to keep checking until a termination condition
    while (1) {

        // Pause for a while before checking again
        sleep(check_period);

        int busy_ports = 0;  // Count of busy ports

        // Parallel region to compute the total count of busy ports.
        #pragma omp parallel for reduction(+:busy_ports)
        for (int i = 0; i < k; i++) {
            busy_ports += port_status[i];
        }

        // If we have hit the threshold of busy ports
        if (busy_ports >= threshold) {
            const int RESPONSE_TAG = 0;

            // Fetch ranks of immediate neighbors
            int nodeRowUp, nodeRowBot, nodeColLeft, nodeColRight;
            MPI_Cart_shift(shared_comm, SHIFT_ROW, DISPLACEMENT, &nodeRowBot, &nodeRowUp);
            MPI_Cart_shift(shared_comm, SHIFT_COL, DISPLACEMENT, &nodeColLeft, &nodeColRight);

            int all_neighbour_ranks[MAX_NEIGHBOURS] = {nodeRowBot, nodeRowUp, nodeColLeft, nodeColRight};
            MPI_Request neighbour_send_req[MAX_NEIGHBOURS];
            MPI_Request neighbour_receive_req[MAX_NEIGHBOURS];
            MPI_Status neighbour_status[MAX_NEIGHBOURS];
            int received_availability[MAX_NEIGHBOURS];
            int available_ports = k - busy_ports;  // Compute available ports

            // Initiate non-blocking communication with neighbors about availability
            for (int i = 0; i < MAX_NEIGHBOURS; i++) {
                int neighbour_rank = all_neighbour_ranks[i];
                if (neighbour_rank != MPI_PROC_NULL) {  
                    MPI_Isend(&available_ports, 1, MPI_INT, neighbour_rank, RESPONSE_TAG, shared_comm, &neighbour_send_req[i]);
                    message_count++;
                    MPI_Irecv(&received_availability[i], 1, MPI_INT, neighbour_rank, RESPONSE_TAG, shared_comm, &neighbour_receive_req[i]);
                }
            }

            // Wait for the communication to complete
            for (int i = 0; i < MAX_NEIGHBOURS; i++) {
                if (all_neighbour_ranks[i] != MPI_PROC_NULL) {
                    MPI_Wait(&neighbour_send_req[i], MPI_STATUS_IGNORE);
                    MPI_Wait(&neighbour_receive_req[i], &neighbour_status[i]);
                }
            }

            // Check the availability of the neighbors
            int all_neighbours_busy = 1;
            for (int i = 0; i < MAX_NEIGHBOURS; i++) {
                if (all_neighbour_ranks[i] != MPI_PROC_NULL) {
                    if (received_availability[i] > 2) {  // threshold is 2
                        all_neighbours_busy = 0;
                    }
                }
            }

            // Fetch the current time for logging
            time_t current_time;
            struct tm * time_info;
            char time_str[20];
            time(&current_time);
            time_info = localtime(&current_time);
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", time_info);

            // Fetch extended neighbors (neighbors that are 2 steps away in the grid)
            int nodeRowUp2, nodeRowBot2, nodeColLeft2, nodeColRight2;
            MPI_Cart_shift(shared_comm, SHIFT_ROW, 2*DISPLACEMENT, &nodeRowBot2, &nodeRowUp2);
            MPI_Cart_shift(shared_comm, SHIFT_COL, 2*DISPLACEMENT, &nodeColLeft2, &nodeColRight2);
            int all_extended_neighbour_ranks[MAX_NEIGHBOURS] = {nodeRowBot2, nodeRowUp2, nodeColLeft2, nodeColRight2};

            // Construct an alert message to send to the base station
            struct {
                int alert_signal;
                char timestamp[20];
                int msg_count;
                int open_ports;
                int coords[2];
                NodeInfo neighbours[MAX_NEIGHBOURS];
                NodeInfo extended_neighbours[MAX_NEIGHBOURS];
            } alertMessage = {1, "", message_count, available_ports};

            // Populate the coordinates
            alertMessage.coords[0] = coord[0];
            alertMessage.coords[1] = coord[1];

            // Populate information about extended neighbors in the alert message
            for (int i = 0; i < MAX_NEIGHBOURS; i++) {
                if (all_extended_neighbour_ranks[i] != MPI_PROC_NULL) {
                    alertMessage.extended_neighbours[i].rank = all_extended_neighbour_ranks[i];
                    MPI_Cart_coords(shared_comm, all_extended_neighbour_ranks[i], shared_ndims, alertMessage.extended_neighbours[i].coords);
                    alertMessage.extended_neighbours[i].available_ports = received_availability[i];
                } else {
                    alertMessage.extended_neighbours[i].rank = MPI_PROC_NULL;
                }
            }

            // Populate information about immediate neighbors in the alert message
            for (int i = 0; i < MAX_NEIGHBOURS; i++) {
                if (all_neighbour_ranks[i] != MPI_PROC_NULL) {
                    alertMessage.neighbours[i].rank = all_neighbour_ranks[i];
                    MPI_Cart_coords(shared_comm, all_neighbour_ranks[i], shared_ndims, alertMessage.neighbours[i].coords);
                    alertMessage.neighbours[i].available_ports = received_availability[i];
                } else {
                    alertMessage.neighbours[i].rank = MPI_PROC_NULL;
                }
            }

            // If all neighbors and the current node are heavily utilized, notify the base station.
            if (all_neighbours_busy) {
                int world_size;
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);
                int base_station_rank = world_size - 1;  // Assuming base station is the last rank

                // Update timestamp in the message
                strncpy(alertMessage.timestamp, time_str, sizeof(time_str));

                // Send alert message to the base station
                MPI_Send(&alertMessage, sizeof(alertMessage), MPI_BYTE, base_station_rank, 99, MPI_COMM_WORLD);
                printf("Rank %d: Alerting base station that this node and its quadrant are fully utilized.\n", rank);

                // Receive a response from the base station
                char base_station_message[100];
                MPI_Recv(base_station_message, sizeof(base_station_message), MPI_CHAR, base_station_rank, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Node %d received from base station: %s\n", rank, base_station_message);
            }
        }
    }

    // Ensure any remaining output is written to the stdout stream
    fflush(stdout);

    // Set the global termination flag and cleanup resources
    *global_terminate_flag = 1;
    free(port_status);

    // Finalize MPI and exit
    MPI_Finalize();
    exit(0);
}



/**
 * Simulates the operation of charging ports on a node.
 * Each port can be either busy or free, and its status changes randomly.
 * This function simulates this randomness for a given node (rank).
 *
 * @param rank                 The rank of the current node in the MPI setup.
 * @param k                    Number of ports in the node.
 * @param port_status          An array of size 'k', representing the status of each port (1 for busy, 0 for free).
 * @param global_terminate_flag A pointer to a flag indicating if the simulation should terminate.
 */
void simulate_charging_ports(int rank, int k, int* port_status, int* global_terminate_flag) {
    
    // Seed the random number generator uniquely for each rank
    // This ensures that different nodes produce different random sequences.
    srand(time(NULL) + rank);

    // Continuously simulate the port operations until a termination signal is received.
    while(!*global_terminate_flag) {

        // Run the simulation in parallel for all ports.
        // Using OpenMP to spawn 'k' threads, one for each port.
        #pragma omp parallel for num_threads(k)
        for(int i = 0; i < k; i++) {
            
            // Simulate a random operation time for the port by making the thread sleep.
            // This sleep duration is a random value between 1 and 3 seconds.
            sleep(rand() % 3 + 1);

            // Access the shared port_status array in a thread-safe manner using critical section.
            // This ensures that only one thread updates the port status at a time.
            #pragma omp critical
            {
                // Randomly determine the status of the port.
                // There's a 1 in 4 chance that the port is busy.
                if(rand() % 4 == 0) {
                    port_status[i] = 1;
                } else {
                    port_status[i] = 0;
                }

                // Uncomment the below line if you wish to print the status of each port.
                // This prints the node rank, port number, its associated thread, and its status.
                // printf("Rank %d, Port %d (Thread %d): Status = %d\n", rank, i, omp_get_thread_num(), port_status[i]);
            }

            // Sleep for 2 seconds to simulate an operational cycle.
            // This is just to slow down the loop and make the simulation more realistic.
            sleep(2);
        }
    }

    // Once the termination signal is received, print a message indicating the node is exiting.
    printf("Rank %d: Received termination signal. Exiting...\n", rank);
    
    // Ensure any remaining output is written to the stdout stream.
    fflush(stdout);

    // Update the global termination flag to signal that this node is exiting.
    *global_terminate_flag = 1;
    
    // Free the memory allocated for port_status array.
    free(port_status);

    // Finalize the MPI environment and exit the process.
    MPI_Finalize();
    exit(0);
}



/**
 * Simulates an EV (Electric Vehicle) charging node's operations and its communication with neighboring nodes.
 * The function sets up the initial conditions of the node (e.g., number of ports, their statuses) and then simulates
 * the charging operations and inter-node communications concurrently using parallel sections.
 * The node awaits a termination message from a base station to safely terminate its operations.
 *
 * @param rank        The rank of the current node in the MPI environment.
 * @param ndims       The number of dimensions in the grid topology of nodes.
 * @param grid_comm   The MPI communicator associated with a grid of nodes.
 */
void simulate_ev_charging_node(int rank, int ndims, MPI_Comm grid_comm) {
    int my_coords[ndims];

    // Retrieve the coordinates of the current process/node in the grid topology
    MPI_Cart_coords(grid_comm, rank, ndims, my_coords);
    printf("Process %d has coordinates (%d, %d)\n", rank, my_coords[0], my_coords[1]);

    int k = 4; // Number of ports
    int* port_status = (int*) malloc(k * sizeof(int));  // Allocate memory for port status

    // Initialize all ports as free
    for(int i = 0; i < k; i++) {
        port_status[i] = 0;
    }

    int global_terminate = 0;  // Flag to indicate global termination

    omp_set_nested(1);  // Enable nested parallelism
    #pragma omp parallel sections
    {
        // Simulate the charging operations of the ports in parallel
        #pragma omp section
        {
            simulate_charging_ports(rank, k, port_status, &global_terminate);
        }

        // Handle the communication with neighboring nodes in parallel
        #pragma omp section
        {
            communicate_nodes(k, port_status, rank, grid_comm, ndims, &global_terminate);
        }
    }

    int terminate = 0;  // Local flag to indicate termination
    MPI_Status status;  // MPI status for receiving messages
    int termination_message_tag = 101; // Custom tag for termination message
    while (!terminate) {
        // Check for termination message from the base station
        MPI_Recv(&terminate, 1, MPI_INT, MPI_ANY_SOURCE, termination_message_tag, grid_comm, &status);
    }

    // Cleanup tasks: flush standard output, free allocated memory, and finalize MPI
    fflush(stdout);
    free(port_status);
    MPI_Finalize();
    exit(0);
}


/**
 * Initializes a 2D grid topology for simulating a charging grid.
 * This function takes the existing MPI communicator and creates a new communicator 
 * with a 2D Cartesian topology based on the specified dimensions.
 * The grid is non-periodic and retains the original ordering of processes.
 *
 * @param size          The number of processes in the existing communicator.
 * @param rank          The rank of the current process in the existing communicator.
 * @param ndims         The number of dimensions in the grid topology (2 for a 2D grid).
 * @param dims          An array specifying the dimensions of the grid (number of rows and columns).
 * @param existing_comm The existing MPI communicator.
 * @param new_comm      A pointer to the new MPI communicator to be created for the grid topology.
 */
void initialise_charging_grid(int size, int rank, int ndims, int *dims, MPI_Comm existing_comm, MPI_Comm *new_comm) {
    int periods[2] = {0, 0}; // Specifies a non-periodic grid (no wrap around)
    int reorder = 0;         // Indicates to preserve the original ordering of processes

    // Print grid information from the master process (rank 0)
    if (rank == 0)
        printf("Comm Size: %d: Grid Dimension =[%d x %d] \n", size, dims[0], dims[1]);

    // Create a new MPI communicator with a 2D Cartesian topology based on the provided dimensions
    MPI_Cart_create(existing_comm, ndims, dims, periods, reorder, new_comm);
}

