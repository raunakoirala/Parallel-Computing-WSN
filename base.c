#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "header_file.h"

#define CHECK_PERIOD 10 // Predefined time period for checking
extern NodeReport *node_reports; // External pointer to an array of node reports

/**
 * Function to check the status of neighbouring nodes and notify the reporting node.
 * This function examines each neighbour's last report time and determines if any neighbour 
 * hasn't reported within a certain threshold. If so, it deems that neighbour as "available" 
 * and notifies the reporting node with a message containing information about available neighbours.
 *
 * @param reporting_node_rank  The rank of the node reporting the status.
 * @param extended_neighbours  An array of neighbour nodes' information.
 * @param base_rank            The rank of the base station.
 */
void check_and_notify_neighbours(int reporting_node_rank, NodeInfo* extended_neighbours, int base_rank) {
    char message[256] = ""; // Buffer for constructing the notification message
    int available = 0;      // A flag indicating if any available neighbour was found

    // Open a log file associated with the base station in append mode
    FILE* file = fopen("base_station_log.txt", "a");
    if (file == NULL) {
        perror("Error opening file for writing");
        return;
    }

    // Log the message indicating the start of availability check
    fprintf(file, "Available station nearby (no report received in last 3 iteration): ");

    // Iterate over the neighbours to determine their report status
    for (int i = 0; i < MAX_NEIGHBOURS && extended_neighbours[i].rank != MPI_PROC_NULL; i++) {
        int neighbour_rank = extended_neighbours[i].rank;
        
        // Calculate the time elapsed since the neighbour's last report
        double elapsed_time = difftime(time(NULL), node_reports[neighbour_rank].last_report_time);
        
        // If a neighbour hasn't reported within 3 times the CHECK_PERIOD, consider it available
        if (elapsed_time > (3 * CHECK_PERIOD)) {
            char temp[50];
            snprintf(temp, sizeof(temp), "Node %d is available. ", neighbour_rank);
            fprintf(file, "%d\n\n", neighbour_rank); // Log the available neighbour's rank
            strcat(message, temp); // Append the message for the reporting node
            available = 1; // Set the availability flag
        }
    }

    // If no available neighbours were found, construct an appropriate message
    int tag = 100; // Tag for MPI communication
    if (!available) {
        strcpy(message, "No available nodes nearby.");
        fprintf(file, "None\n\n"); // Log the lack of available neighbours
    }

    // Send the constructed message back to the reporting node
    MPI_Send(message, strlen(message) + 1, MPI_CHAR, reporting_node_rank, tag, MPI_COMM_WORLD);

    // Close the opened log file
    fclose(file);
}


/**
 * Function to log various details pertaining to a node's status and its neighbours to a file.
 * The logged details include timestamps, reporting node's status, adjacent and nearby node details,
 * and the number of messages exchanged between the reporting node and the base station.
 *
 * @param source               The rank of the reporting node.
 * @param timestamp            The timestamp when the alert was reported.
 * @param msg_count            The count of messages sent between the reporting node and the base station.
 * @param iteration            Current iteration of the simulation.
 * @param open_ports           Number of available ports in the reporting node.
 * @param neighbours           An array of the reporting node's direct neighbours' information.
 * @param extended_neighbours  An array of the reporting node's extended (or nearby) neighbours' information.
 * @param coords               Coordinates of the reporting node in the grid.
 * @param total_comm_time      The total communication time.
 */
void log_to_file(int source, const char* timestamp, int msg_count, int iteration, int open_ports, NodeInfo* neighbours, NodeInfo* extended_neighbours, int* coords, float total_comm_time) {

    // Open the base station's log file in append mode
    FILE* file = fopen("base_station_log.txt", "a");
    if (file == NULL) {
        perror("Error opening file for writing");
        return;
    }

    // Retrieve the current time and format it as a string
    char logged_time[20];
    time_t rawtime;
    struct tm * timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(logged_time, sizeof(logged_time), "%Y-%m-%d %H:%M:%S", timeinfo);

    // Logging the iteration, logged time and alert time to the file
    fprintf(file, "------------------------------------------------------------------------------------------------------------\n");
    fprintf(file, "Iteration : %d\nLogged time : %s\nAlert reported time : %s\n\n", iteration, logged_time, timestamp);


    // Log details of the adjacent nodes
    int adj_nodes = 0;
    char ipv4[16];  // buffer to hold the generated IPv4 address

    // For the source node:
    sprintf(ipv4, "192.168.0.%d", source);
    fprintf(file, "Reporting Nodes\t\tCoord\t\tPort Value\tAvailable Port\tIPv4\n");
    fprintf(file, "%d\t\t\t(%d, %d)\t\t4\t\t%d\t\t%s\n\n", source, coords[0], coords[1], open_ports, ipv4);

    // For the adjacent nodes:
    fprintf(file, "Adjacent Nodes\t\tCoord\t\tPort Value\tAvailable Port\tIPv4\n");
    for (int i = 0; i < MAX_NEIGHBOURS; i++) {
        if (neighbours[i].rank != MPI_PROC_NULL) {
            sprintf(ipv4, "192.168.0.%d", neighbours[i].rank);
            fprintf(file, "%d\t\t\t(%d, %d)\t\t4\t\t%d\t\t%s\n\n", neighbours[i].rank, neighbours[i].coords[0], neighbours[i].coords[1], neighbours[i].available_ports, ipv4);
            adj_nodes++;
        }
    }

    // Additional details about the adjacent nodes
    fprintf(file, "Number of adjacent node: %d\nAvailability to be considered full: 2 or fewer ports are free\n\n", adj_nodes);

    // Log details of the nearby (or extended) nodes
    fprintf(file, "Nearby Nodes\t\tCoord\n");
    for (int i = 0; i < MAX_NEIGHBOURS; i++) {
        if (extended_neighbours[i].rank != MPI_PROC_NULL) {
            fprintf(file, "%d\t\t\t(%d, %d)\n\n", extended_neighbours[i].rank, extended_neighbours[i].coords[0], extended_neighbours[i].coords[1]);
        }
    }

    // Log the total number of messages exchanged
    fprintf(file, "Total Messages send between reporting node and base station: %d\n", msg_count);

    fclose(file); // Close the opened log file
}


/**
 * This is the main function representing the operations of the base station in an electric vehicle
 * charging station simulation. The base station continuously receives messages from different nodes,
 * logs relevant data, sends notifications, and checks for any alerts from the nodes about the utilization
 * status of the charging ports. Once all iterations are completed, it terminates all node processes.
 *
 * @param base_rank       The rank of the base station in the MPI environment.
 * @param world_comm      The main communicator encompassing all MPI processes.
 * @param node_comm       Pointer to the communicator that represents node processes.
 * @param dims            Dimensions of the grid of nodes.
 * @param size            The total number of processes (including the base station).
 * @param iterations      The number of iterations for which the simulation should run.
 */
void base_func(int base_rank, MPI_Comm world_comm, MPI_Comm *node_comm, int *dims, int size, int iterations) {

    int iteration = 0;                  // Current iteration count
    double start_time, total_comm_time; // Variables to measure communication time

    // Struct for the message that will be received from nodes
    struct {
        int alert_signal;                              // Alert signal (1 if there's an alert, 0 otherwise)
        char timestamp[20];                            // Time the message was sent
        int msg_count;                                 // Message count
        int open_ports;                                // Number of open charging ports
        int coords[2];                                 // Coordinates of the sending node
        NodeInfo neighbours[MAX_NEIGHBOURS];           // Info about the neighbours of the sending node
        NodeInfo extended_neighbours[MAX_NEIGHBOURS];  // Info about the extended neighbours of the sending node
    } receivedMessage;

    MPI_Status status;  // MPI status to retrieve source and other info from received messages

    while(1) {
        // Receive a message from any node
        start_time = MPI_Wtime();
        MPI_Recv(&receivedMessage, sizeof(receivedMessage), MPI_BYTE, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, &status);
        total_comm_time = MPI_Wtime() - start_time;

        // Update the reporting status and time for the sending node
        node_reports[status.MPI_SOURCE].has_reported = 1;
        node_reports[status.MPI_SOURCE].last_report_time = time(NULL);

        // If an alert signal is received from a node
        if (receivedMessage.alert_signal == 1) {
            printf("Base Station: Received alert from node %d,quadrant is fully utilized.\n Timestamp: %s\n Message Count: %d\n", status.MPI_SOURCE, receivedMessage.timestamp, receivedMessage.msg_count);
            
            // Log the details of the received message to a file
            log_to_file(status.MPI_SOURCE, receivedMessage.timestamp, receivedMessage.msg_count, iteration, receivedMessage.open_ports, receivedMessage.neighbours, receivedMessage.extended_neighbours,
            receivedMessage.coords, total_comm_time);
            
            // Check the status of the extended neighbours and notify if needed
            check_and_notify_neighbours(status.MPI_SOURCE, receivedMessage.extended_neighbours, base_rank);
        }

        iteration++;


        // If all iterations are done, send a termination signal to all nodes
        if (iteration == iterations) {
            for (int i = 0; i < size - 1; i++) {
                int terminate = 1;
                MPI_Send(&terminate, 1, MPI_INT, i, 101, MPI_COMM_WORLD);
            }
            break;
        }
    }

    sleep(2); // Allow some time for all pending MPI messages to be processed

    printf("Base station process has ended: ran for %d iterations!\n", iterations);
    fflush(stdout);
    MPI_Abort(world_comm, 0); // Abort the MPI environment
}

