#include <BaseClass.h>
#include <iostream>
#include <mpi.h>
#include <vector>

#pragma once

namespace ttk {

  class BaseMPIClass : public BaseClass {

  public:
    MPI_Comm MPIComm;
    int GlobalElementToCompute;
    long int *GlobalIdsArray;
    unsigned char *PointGhostArray;
    unsigned char *CellGhostArray;
    int *ProcessId;
    MPI_Datatype MessageType;
    int MyRank;
    int NumberOfProcesses;
    std::vector<std::vector<int>> Vertex2Process;
    int *IsOnMPIBoundary;
    int vertexNumber;

    BaseMPIClass();

    virtual ~BaseMPIClass() = default;

    void setMyRank(int rank) {
      this->MyRank = rank;
    }

    void setNumberOfProcesses(int number) {
      this->NumberOfProcesses = number;
    }

    void setMPIComm(MPI_Comm comm) {
      this->MPIComm = comm;
    }

    void setGlobalElementToCompute(int number) {
      this->GlobalElementToCompute = number;
    }

    void setGlobalIdsArray(long int *array) {
      this->GlobalIdsArray = array;
    }

    void setPointGhostArray(unsigned char *array) {
      this->PointGhostArray = array;
    }

    void setCellGhostArray(unsigned char *array) {
      this->CellGhostArray = array;
    }

    void setProcessId(int *processId) {
      this->ProcessId = processId;
    }

    void setVertex2Process(std::vector<std::vector<int>> array) {
      this->Vertex2Process = array;
    }

    void setIsOnMPIBoundary(int *array) {
      this->IsOnMPIBoundary = array;
    }

    // void setFinishedElement(int number) {
    //   this->finishedElement = number;
    // }

    // void setKeepWorking(bool number) {
    //   this->keepWorking = number;
    // }

    // void setTaskCounter(int number) {
    //   this->taskCounter = number;
    // }

    // void setGlobalElementCounter(int number) {
    //   this->globalElementCounter = number;
    // }
  };
} // namespace ttk
