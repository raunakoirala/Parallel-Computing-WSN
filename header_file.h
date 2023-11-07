#ifndef HEADER_H
#define HEADER_H

#include <mpi.h>
#include <omp.h>
#include <string.h>

#define MAX_NEIGHBOURS 4
#define NDIMS 2
#define CHECK_PERIOD 10 
#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISPLACEMENT 1


typedef struct {
    time_t last_report_time;
    int has_reported;
} NodeReport;

typedef struct {
    int rank;
    int coords[2];
    int available_ports;
} NodeInfo;


void communicate_nodes(int k, int* port_status, int rank, MPI_Comm shared_comm, int shared_ndims, int* global_terminate_flag) ;
void simulate_charging_ports(int rank, int k, int* port_status,int* global_terminate_flag);
void simulate_ev_charging_node(int rank, int ndims, MPI_Comm grid_comm);
void initialise_charging_grid(int size, int rank, int ndims, int *dims, MPI_Comm existing_comm, MPI_Comm *new_comm);

void check_and_notify_neighbours(int reporting_node_rank, NodeInfo* extended_neighbours, int base_rank);
void log_to_file(int source, const char* timestamp, int msg_count, int iteration, int open_ports, NodeInfo* neighbours, NodeInfo* extended_neighbours, int* coords, float total_comm_time);
void base_func(int base_rank, MPI_Comm world_comm, MPI_Comm *node_comm, int *dims, int size, int iterations);



#endif