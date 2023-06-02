#include <iostream>
#include <pthread.h>
#include <cmath>
#include <queue>
#include <mpi.h>
#include <memory.h>
#include <cstdlib>
#include <csignal>

#define L 10000
#define LISTS_COUNT 1
#define TASK_COUNT 787
#define MIN_TASKS_TO_SHARE 1

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
int sendingTasks[TASK_COUNT];
std::queue<int> tasksQ = std::queue<int>();

double summaryDisbalance = 0;
bool finishedExecution = false;

int processCount;
int processRank;
int remainingTasks;
int executedTasks;
int executedTasksCurrent;
int additionalCount;
double globalRes = 0;

long startWeightCount = 0;
long endExecutedWeight = 0;

void executeTaskSet() {
    int weight;
    pthread_mutex_lock(&mutex);
    while (!tasksQ.empty()) {
        weight = tasksQ.front();
        tasksQ.pop();
        pthread_mutex_unlock(&mutex);

        for (int j = 0; j < weight; j++) {
            globalRes += cos(0.001234);
        }

        executedTasksCurrent++;
        endExecutedWeight += weight;
        pthread_mutex_lock(&mutex);
    }
    pthread_mutex_unlock(&mutex);
}

void fillTasks(int iterCounter) {
        for (int i = 0; i < TASK_COUNT; i++) {
        tasksQ.push(abs(50 - i % 100) * abs(processRank - (iterCounter % processCount)) * L);
        startWeightCount += tasksQ.back();
    }
}

void addTasks(const int count, const int *task) {
    for (int i = 0; i < count; ++i) {
        tasksQ.push(task[i]);
    }
}

void fillTasksToSend(int count) {
    for (int i = 0; i < count; ++i) {
        sendingTasks[i] = tasksQ.front();
        tasksQ.pop();
    }
}


void clearQueue(){
    while(!tasksQ.empty()){
    	tasksQ.pop();
    }
}

void *ExecutorStartRoutine(void *args) {

    double startTime;
    double finishTime;
    double iterationDuration;
    double shortestIteration;
    double longestIteration;
    int threadResponse;

    for (int i = 0; i < LISTS_COUNT; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        startTime = MPI_Wtime();

        pthread_mutex_lock(&mutex);
	clearQueue();
	pthread_mutex_unlock(&mutex);

        executedTasks = 0;
        executedTasksCurrent = 0;

        additionalCount = 0;

        std::cout << ANSI_RED << "[" << processRank << "E] Iteration " << i << ". Initializing sendingTasks. "
                  << ANSI_RESET
                  << std::endl;

        pthread_mutex_lock(&mutex);
        fillTasks(i);
        pthread_mutex_unlock(&mutex);

        executeTaskSet();
        executedTasks += executedTasksCurrent;

        std::cout << ANSI_BLUE << "[" << processRank << "E] Process executed sendingTasks in " <<
                  MPI_Wtime() - startTime << " Now requesting for some additional. " << ANSI_RESET << std::endl;

        for (int procIdx = 0; procIdx < processCount; procIdx++) {

            if (procIdx != processRank) {
                std::cout << ANSI_WHITE << "[" << processRank << "E] Process is asking " << procIdx <<
                          " for some sendingTasks." << ANSI_RESET << std::endl;

                MPI_Send(&processRank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);

                std::cout << ANSI_PURPLE << "[" << processRank << "E] waiting for task count" << ANSI_RESET
                          << std::endl;

                MPI_Recv(&threadResponse, 1, MPI_INT, procIdx, SENDING_TASK_COUNT_TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                std::cout << ANSI_WHITE << "[" << processRank << "E] Process " << procIdx << " answered "
                          << threadResponse << ANSI_RESET << std::endl;

                if (threadResponse != NO_TASKS_TO_SHARE) {
                    additionalCount = threadResponse;

                    int addingTasks[TASK_COUNT];

                    std::cout << ANSI_PURPLE << "[" << processRank << "E] waiting for sendingTasks" << ANSI_RESET
                              << std::endl;

                    MPI_Recv(addingTasks, additionalCount, MPI_INT, procIdx, SENDING_TASKS_TAG, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);

                    pthread_mutex_lock(&mutex);
                    addTasks(additionalCount, addingTasks);
                    pthread_mutex_unlock(&mutex);

                    executeTaskSet();
                    executedTasks += executedTasksCurrent;
                }
            }
        }

        finishTime = MPI_Wtime();
        iterationDuration = finishTime - startTime;

        MPI_Allreduce(&iterationDuration, &longestIteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iterationDuration, &shortestIteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);


        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << ANSI_GREEN << "[" << processRank << "E] executed " << executedTasks <<
                  " sendingTasks. " << additionalCount << " were additional." << ANSI_RESET << std::endl;
        std::cout << ANSI_CYAN << "[" << processRank << "E] Cos sum is " << globalRes << ". Time taken: "
                  << iterationDuration << std::endl;

        summaryDisbalance += (longestIteration - shortestIteration) / longestIteration;

        std::cout << "[" << processRank << "E] Max time difference: " << longestIteration - shortestIteration
                  << ANSI_RESET << std::endl;
        std::cout << ANSI_PURPLE_BG << "[" << processRank << "E] Disbalance rate is " <<
                  ((longestIteration - shortestIteration) / longestIteration) * 100 << "%" << ANSI_RESET << std::endl;


        long weightSum = 0;
        long weightCountSum = 0;

        MPI_Allreduce(&endExecutedWeight, &weightSum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&startWeightCount, &weightCountSum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        std::cout << ANSI_PURPLE_BG << "[" << processRank << "E] Tasks weight summary executed: " << weightSum
                  << ANSI_RESET << std::endl;
        std::cout << ANSI_PURPLE_BG << "[" << processRank << "E] Tasks weight on start iteration: " << weightCountSum
                  << ANSI_RESET << std::endl;
    }

    std::cout << ANSI_RED << "[" << processRank << "E] Finished iterations, sending signal" << ANSI_RESET << std::endl;

    pthread_mutex_lock(&mutex);
    finishedExecution = true;
    pthread_mutex_unlock(&mutex);

    int signal = EXECUTOR_FINISHED_WORK;

    MPI_Send(&signal, 1, MPI_INT, processRank, 888, MPI_COMM_WORLD);


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

        std::cout << ANSI_YELLOW << "[" << processRank << "R] Process " << askingProcRank
                  << " requested sendingTasks. I have "
                  << tasksQ.size() << " sendingTasks now. " << ANSI_RESET << std::endl;

        pthread_mutex_lock(&mutex);
        if (!tasksQ.empty()) {
            answer = (tasksQ.size()) / (3);
            std::cout << answer  << std::endl;

            fillTasksToSend(answer);

            std::cout << ANSI_PURPLE << "[" << processRank << "R] Sharing " << answer << " sendingTasks. " << ANSI_RESET
                      << std::endl;

            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT_TAG, MPI_COMM_WORLD);
            MPI_Send(sendingTasks, answer, MPI_INT, askingProcRank, SENDING_TASKS_TAG,
                     MPI_COMM_WORLD);
        } else {

            answer = NO_TASKS_TO_SHARE;
            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT_TAG, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&mutex);
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
