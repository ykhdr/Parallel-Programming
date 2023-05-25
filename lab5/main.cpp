#include <iostream>
#include <pthread.h>
#include <cmath>
#include <mpi.h>
#include <memory.h>
#include <cstdlib>

#define L 1000
#define LISTS_COUNT 1
#define TASK_COUNT 787
#define MIN_TASKS_TO_SHARE 2

#define EXECUTOR_FINISHED_WORK (-1)
#define NO_TASKS_TO_SHARE (-565)
#define SENDING_TASKS_TAG 778
#define SENDING_TASK_COUNT_TAG 121

#define ANSI_RESET "\033[0m"
#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_BLUE "\033[34m"
#define ANSI_PURPLE "\033[35m"
#define ANSI_CYAN "\033[36m"
#define ANSI_WHITE "\033[37m"
#define ANSI_PURPLE_BG "\033[45m\033[30m"

pthread_t threads[2];
pthread_mutex_t mutex;
int *tasks;

double summaryDisbalance = 0;
bool finishedExecution = false;

int processCount;
int processRank;
int remainingTasks;
int executedTasks;
int additionalTasks;
double globalRes = 0;

void initializeTaskSet(int *taskSet, int taskCount, int iterCounter) {
    for (int i = 0; i < taskCount; i++) {
        taskSet[i] = abs(50 - i % 100) * abs(processRank - (iterCounter % processCount)) * L;
    }
}

void executeTaskSet(const int *taskSet) {
    for (int i = 0; i < remainingTasks; i++) {
        int weight = taskSet[i];

        for (int j = 0; j < weight; j++) {
            globalRes += cos(0.001234);
        }

        executedTasks++;
    }
    remainingTasks = 0;
}

void *ExecutorStartRoutine(void *args) {
    tasks = new int[TASK_COUNT];
    double startTime;
    double finishTime;
    double iterationDuration;
    double shortestIteration;
    double longestIteration;
    int executedTasksSum;
    int threadResponse;

    for (int i = 0; i < LISTS_COUNT; i++) {
        startTime = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << ANSI_RED << "[" << processRank << "E] Iteration " << i << ". Initializing tasks. " << ANSI_RESET
                  << std::endl;
        initializeTaskSet(tasks, TASK_COUNT, i);

        executedTasks = 0;
        remainingTasks = TASK_COUNT;
        additionalTasks = 0;
        executedTasksSum = 0;

        executeTaskSet(tasks);
        std::cout << ANSI_BLUE << "[" << processRank << "E] Process executed tasks in " <<
                  MPI_Wtime() - startTime << " Now requesting for some additional. " << ANSI_RESET << std::endl;


        for (int procIdx = 0; procIdx < processCount; procIdx++) {

            if (procIdx != processRank) {
                std::cout << ANSI_WHITE << "[" << processRank << "E] Process is asking " << procIdx <<
                          " for some tasks." << ANSI_RESET << std::endl;

                MPI_Send(&processRank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);

                std::cout << ANSI_PURPLE << "[" << processRank << "E] waiting for task count" << ANSI_RESET << std::endl;

                MPI_Recv(&threadResponse, 1, MPI_INT, procIdx, SENDING_TASK_COUNT_TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                std::cout << ANSI_WHITE << "[" << processRank << "E] Process " << procIdx << " answered " << threadResponse << ANSI_RESET << std::endl;

                if (threadResponse != NO_TASKS_TO_SHARE) {
                    additionalTasks = threadResponse;
                    memset(tasks, 0, TASK_COUNT);

                    std::cout << ANSI_PURPLE << "[" << processRank << "E] waiting for tasks" << ANSI_RESET << std::endl;

                    MPI_Recv(tasks, additionalTasks, MPI_INT, procIdx, SENDING_TASKS_TAG, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);

                    //pthread_mutex_lock(&mutex);
                    remainingTasks = additionalTasks;
                    //pthread_mutex_unlock(&mutex);

                    executeTaskSet(tasks);
                }
            }
        }

        finishTime = MPI_Wtime();
        iterationDuration = finishTime - startTime;

        MPI_Allreduce(&iterationDuration, &longestIteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iterationDuration, &shortestIteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        MPI_Allreduce(&executedTasks, &executedTasksSum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << ANSI_GREEN << "[" << processRank << "E] executed " << executedTasks <<
                  " tasks. " << additionalTasks << " were additional." << ANSI_RESET << std::endl;
        std::cout << ANSI_CYAN << "[" << processRank << "E] Cos sum is " << globalRes << ". Time taken: "
                  << iterationDuration << std::endl;

        summaryDisbalance += (longestIteration - shortestIteration) / longestIteration;

        std::cout << "[" << processRank << "E] Max time difference: " << longestIteration - shortestIteration
                  << ANSI_RESET << std::endl;
        std::cout << ANSI_PURPLE_BG << "[" << processRank << "E] Disbalance rate is " <<
                  ((longestIteration - shortestIteration) / longestIteration) * 100 << "%" << ANSI_RESET << std::endl;

        std::cout << ANSI_YELLOW << "[" << processRank << "E] Tasks summary executed: " << executedTasksSum << ANSI_RESET << std::endl;
        std::cout << ANSI_YELLOW << "[" << processRank << "E] Tasks on start iteration: " << TASK_COUNT * processCount << ANSI_RESET << std::endl;
    }

    std::cout << ANSI_RED << "[" << processRank << "E] Finished iterations, sending signal" << ANSI_RESET << std::endl;

    //pthread_mutex_lock(&mutex);
    finishedExecution = true;
    //pthread_mutex_unlock(&mutex);

    int signal = EXECUTOR_FINISHED_WORK;

    MPI_Send(&signal, 1, MPI_INT, processRank, 888, MPI_COMM_WORLD);

    delete[] tasks;
    pthread_exit(nullptr);
}

void *ReceiverStartRoutine(void *args) {
    int askingProcRank;
    int answer;
    int pendingMessage;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    while (!finishedExecution) {
        MPI_Recv(&pendingMessage, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD, &status);

        if (pendingMessage == EXECUTOR_FINISHED_WORK) {
            std::cout << ANSI_RED << "[" << processRank << "R] Executor finished work on proc" << ANSI_RESET
                      << std::endl;
            break;
        }

        askingProcRank = pendingMessage;
        //pthread_mutex_lock(&mutex);

        std::cout << ANSI_YELLOW << "[" << processRank << "R] Process " << askingProcRank << " requested tasks. I have "
                  <<
                  remainingTasks << " tasks now. " << ANSI_RESET << std::endl;

        if (remainingTasks >= MIN_TASKS_TO_SHARE) {
            answer = remainingTasks / (processCount * 2);

            std::cout << ANSI_PURPLE << "[" << processRank << "R] Sharing " << answer << " tasks. " << ANSI_RESET
                      << std::endl;

            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT_TAG, MPI_COMM_WORLD);
            MPI_Send(&tasks[remainingTasks - answer], answer, MPI_INT, askingProcRank, SENDING_TASKS_TAG,
                     MPI_COMM_WORLD);
            remainingTasks -= answer;
        } else {
            answer = NO_TASKS_TO_SHARE;
            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT_TAG, MPI_COMM_WORLD);
        }

        //pthread_mutex_unlock(&mutex);
    }

    pthread_exit(nullptr);
}


int main(int argc, char *argv[]) {
    int threadSupport;
    pthread_attr_t threadAttributes;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSupport);
    if (threadSupport != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    pthread_mutex_init(&mutex, nullptr);

    double start = MPI_Wtime();
    pthread_attr_init(&threadAttributes);
    pthread_attr_setdetachstate(&threadAttributes, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &threadAttributes, ReceiverStartRoutine, NULL);
    pthread_create(&threads[1], &threadAttributes, ExecutorStartRoutine, NULL);
    pthread_join(threads[0], nullptr);
    pthread_join(threads[1], nullptr);
    pthread_attr_destroy(&threadAttributes);
    pthread_mutex_destroy(&mutex);

    if (processRank == 0) {
        std::cout << ANSI_GREEN << "Summary disbalance:" << summaryDisbalance / (LISTS_COUNT) * 100 << "%" << ANSI_GREEN
                  << std::endl;
        std::cout << ANSI_GREEN << "time taken: " << MPI_Wtime() - start << ANSI_GREEN << std::endl;
    }

    MPI_Finalize();
    return 0;
}