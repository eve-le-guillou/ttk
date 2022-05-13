/// \ingroup base
/// \class ttk::BaseMPIClass
/// \author Michael Will <mswill@rhrk.uni-kl.de>
/// \author Eve Le Guillou <eve.le-guillou@lip6.fr>
/// \date April 2022
///
/// \brief Base Class and utilities for MPI implementation.

#pragma once
#include <BaseClass.h>
#include <iostream>
#include <vector>

#if TTK_ENABLE_MPI
#include <mpi.h>

namespace ttk {
  COMMON_EXPORTS extern int MPIrank_;

  inline MPI_Datatype getMPIType(const float ttkNotUsed(val)) {
    return MPI_FLOAT;
  };
  inline MPI_Datatype getMPIType(const int ttkNotUsed(val)) {
    return MPI_INT;
  };
  inline MPI_Datatype getMPIType(const unsigned int ttkNotUsed(val)) {
    return MPI_UNSIGNED;
  };
  inline MPI_Datatype getMPIType(const double ttkNotUsed(val)) {
    return MPI_DOUBLE;
  };
  inline MPI_Datatype getMPIType(const long double ttkNotUsed(val)) {
    return MPI_LONG_DOUBLE;
  };
  inline MPI_Datatype getMPIType(const long ttkNotUsed(val)) {
    return MPI_LONG;
  };
  inline MPI_Datatype getMPIType(const unsigned long ttkNotUsed(val)) {
    return MPI_UNSIGNED_LONG;
  };
  inline MPI_Datatype getMPIType(const long long ttkNotUsed(val)) {
    return MPI_LONG_LONG;
  };
  inline MPI_Datatype getMPIType(const unsigned long long ttkNotUsed(val)) {
    return MPI_UNSIGNED_LONG_LONG;
  };

  inline bool isRunningWithMPI() {
    int flag_i;
    MPI_Initialized(&flag_i);
    return flag_i;
  }

  class BaseMPIClass : public BaseClass {

  public:
    BaseMPIClass();
    virtual ~BaseMPIClass() = default;
    MPI_Comm MPIComm;
    int GlobalElementToCompute;
    long int *GlobalIdsArray;
    unsigned char *PointGhostArray;
    int *ProcessId;
    MPI_Datatype MessageType;
    int MyRank;
    int NumberOfProcesses;

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

    void setProcessId(int *processId) {
      this->ProcessId = processId;
    }
  };
} // namespace ttk

#endif
