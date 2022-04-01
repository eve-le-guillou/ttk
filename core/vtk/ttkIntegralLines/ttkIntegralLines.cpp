#include <ttkIntegralLines.h>
#include <ttkMacros.h>
#include <ttkUtils.h>

#include <DataSetAttributes.h>
#include <vtkCellData.h>
#include <vtkCommunicator.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkInformation.h>
#include <vtkMPI.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
#include <vtkMultiProcessController.h>
#include <vtkObjectFactory.h>
#include <vtkPartitionedDataSet.h>
#include <vtkPointData.h>
#include <vtkPointSet.h>
#include <vtkUnstructuredGrid.h>

using namespace std;
using namespace ttk;

vtkStandardNewMacro(ttkIntegralLines)

  ttkIntegralLines::ttkIntegralLines() {
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfOutputPorts(1);
}

#if TTK_ENABLE_MPI
MPI_Comm MPIGetComm() {
  MPI_Comm comm = MPI_COMM_NULL;
  vtkMultiProcessController *controller
    = vtkMultiProcessController::GetGlobalController();
  vtkMPICommunicator *vtkComm
    = vtkMPICommunicator::SafeDownCast(controller->GetCommunicator());
  if(vtkComm) {
    if(vtkComm->GetMPIComm()) {
      comm = *(vtkComm->GetMPIComm()->GetHandle());
    }
  }

  return comm;
}
#endif

ttkIntegralLines::~ttkIntegralLines() = default;

int ttkIntegralLines::FillInputPortInformation(int port, vtkInformation *info) {
  if(port == 0)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkDataSet");
  if(port == 1)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPointSet");

  return 1;
}

int ttkIntegralLines::FillOutputPortInformation(int port,
                                                vtkInformation *info) {
  if(port == 0)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkUnstructuredGrid");

  return 1;
}

int ttkIntegralLines::getTrajectories(
  vtkDataSet *input,
  ttk::Triangulation *triangulation,
  LinkedList<std::vector<SimplexId> *, TABULAR_SIZE> &trajectories,
  LinkedList<std::vector<double> *, TABULAR_SIZE> &distancesFromSeed,
  LinkedList<int, TABULAR_SIZE> &seedIdentifiers,
  vtkUnstructuredGrid *output) {
  vtkSmartPointer<vtkUnstructuredGrid> ug
    = vtkSmartPointer<vtkUnstructuredGrid>::New();
  vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkFloatArray> dist = vtkSmartPointer<vtkFloatArray>::New();
  vtkSmartPointer<vtkFloatArray> identifier
    = vtkSmartPointer<vtkFloatArray>::New();
  dist->SetNumberOfComponents(1);
  dist->SetName("DistanceFromSeed");
  identifier->SetNumberOfComponents(1);
  identifier->SetName("SeedIdentifier");

  // here, copy the original scalars
  int numberOfArrays = input->GetPointData()->GetNumberOfArrays();

  vector<vtkDataArray *> scalarArrays;
  for(int k = 0; k < numberOfArrays; ++k) {
    auto a = input->GetPointData()->GetArray(k);

    if(a->GetNumberOfComponents() == 1)
      scalarArrays.push_back(a);
  }
  // not efficient, implicit conversion to double
  vector<vtkSmartPointer<vtkDoubleArray>> inputScalars(scalarArrays.size());
  for(unsigned int k = 0; k < scalarArrays.size(); ++k) {
    inputScalars[k] = vtkSmartPointer<vtkDoubleArray>::New();
    inputScalars[k]->SetNumberOfComponents(1);
    inputScalars[k]->SetName(scalarArrays[k]->GetName());
  }

  float p[3];
  vtkIdType ids[2];
  ListNode<std::vector<int> *, TABULAR_SIZE> *trajectory
    = trajectories.firstTab;
  ListNode<std::vector<double> *, TABULAR_SIZE> *distanceFromSeed
    = distancesFromSeed.firstTab;
  ListNode<int, TABULAR_SIZE> *seedIdentifier = seedIdentifiers.firstTab;
  // for(SimplexId i = 0; i < trajectories.sizeList; ++i) {
  int arrayLength = 0;
  while(trajectory != NULL) {
    if(trajectory->nextTab == NULL) {
      arrayLength = trajectories.numberOfElement % TABULAR_SIZE;
    } else {
      arrayLength = TABULAR_SIZE;
    }
    for(int i = 0; i < arrayLength; i++) {
      if(trajectory->tab[i].size()) {
        SimplexId vertex = trajectory->tab->at(i)->at(0);
        // init
        triangulation->getVertexPoint(vertex, p[0], p[1], p[2]);
        ids[0] = pts->InsertNextPoint(p);
        // distanceScalars
        dist->InsertNextTuple1(distanceFromSeed->tab->at(i)->at(0));
        identifier->InsertNextTuple1(seedIdentifier->tab->at(i));
        // inputScalars
        for(unsigned int k = 0; k < scalarArrays.size(); ++k)
          inputScalars[k]->InsertNextTuple1(scalarArrays[k]->GetTuple1(vertex));
        for(SimplexId j = 1; j < (SimplexId)(trajectory->tab->at(i)->size());
            ++j) {
          vertex = trajectory->tab->at(i)->at(j);
          triangulation->getVertexPoint(vertex, p[0], p[1], p[2]);
          ids[1] = pts->InsertNextPoint(p);
          // distanceScalars
          dist->InsertNextTuple1(distanceFromSeed->tab->at(i)->at(j));
          identifier->InsertNextTuple1(seedIdentifier->tab->at(i));

          // inputScalars
          for(unsigned int k = 0; k < scalarArrays.size(); ++k)
            inputScalars[k]->InsertNextTuple1(
              scalarArrays[k]->GetTuple1(vertex));

          ug->InsertNextCell(VTK_LINE, 2, ids);

          // iteration
          ids[0] = ids[1];
        }
        delete(trajectory->tab->at(i));
        delete(distanceFromSeed->tab->at(i));
      }
    }
    trajectory = trajectory->nextTab;
    distanceFromSeed = distanceFromSeed->nextTab;
    seedIdentifier = seedIdentifier->nextTab;
  }
  ug->SetPoints(pts);
  ug->GetPointData()->AddArray(dist);
  ug->GetPointData()->AddArray(identifier);
  for(unsigned int k = 0; k < scalarArrays.size(); ++k)
    ug->GetPointData()->AddArray(inputScalars[k]);

  output->ShallowCopy(ug);

  return 0;
}

int ttkIntegralLines::RequestData(vtkInformation *ttkNotUsed(request),
                                  vtkInformationVector **inputVector,
                                  vtkInformationVector *outputVector) {
#if TTK_ENABLE_MPI
  // Get processes information
  vtkMPIController *controller = vtkMPIController::SafeDownCast(
    vtkMultiProcessController::GetGlobalController());
  controller->Barrier();
  int myRank = controller->GetLocalProcessId();
  int numberOfProcesses = controller->GetNumberOfProcesses();
  MPI_Comm comm = MPI_COMM_NULL;
  comm = MPIGetComm();
  this->setMPIComm(comm);
  this->setNumberOfProcesses(numberOfProcesses);
  this->setMyRank(myRank);
#endif

  vtkDataSet *domain = vtkDataSet::GetData(inputVector[0], 0);
  vtkPointSet *seeds = vtkPointSet::GetData(inputVector[1], 0);
  vtkUnstructuredGrid *output = vtkUnstructuredGrid::GetData(outputVector, 0);

  ttk::Triangulation *triangulation = ttkAlgorithm::GetTriangulation(domain);
  vtkDataArray *inputScalars = this->GetInputArrayToProcess(0, domain);

  vtkDataArray *inputOffsets
    = this->GetOrderArray(domain, 0, 1, ForceInputOffsetScalarField);

  const SimplexId numberOfPointsInDomain = domain->GetNumberOfPoints();
  this->setVertexNumber(numberOfPointsInDomain);
  printMsg("number of points in domain"
           + std::to_string(numberOfPointsInDomain));
  SimplexId numberOfPointsInSeeds = seeds->GetNumberOfPoints();
  SimplexId *inputIdentifiers;
#if TTK_ENABLE_MPI
  vtkSmartPointer<vtkIntArray> vtkInputIdentifiers
    = vtkSmartPointer<vtkIntArray>::New();
  vtkInputIdentifiers->SetNumberOfComponents(1);
  vtkInputIdentifiers->SetNumberOfTuples(0);
  unsigned char *pointGhostArray = static_cast<unsigned char *>(
    ttkUtils::GetVoidPointer(domain->GetPointGhostArray()));
  this->setPointGhostArray(pointGhostArray);
  int *processId = static_cast<int *>(
    ttkUtils::GetVoidPointer(domain->GetPointData()->GetArray("ProcessId")));

  long int *globalPointsId = static_cast<long int *>(ttkUtils::GetVoidPointer(
    domain->GetPointData()->GetArray("GlobalPointIds")));
  this->setGlobalIdsArray(globalPointsId);
  this->setProcessId(processId);

  if(this->NumberOfProcesses > 1) {
    int totalSeeds;
    controller->Reduce(
      &numberOfPointsInSeeds, &totalSeeds, 1, vtkCommunicator::SUM_OP, 0);
    int isDistributed;

    if(myRank == 0) {
      isDistributed = numberOfPointsInSeeds != totalSeeds;
    }

    controller->Broadcast(&isDistributed, 1, 0);

    if(!isDistributed) {
      vtkDataArray *globalSeedsId;
      if(myRank == 0) {
        globalSeedsId = seeds->GetPointData()->GetArray("GlobalPointIds");
      } else {
        globalSeedsId = vtkDataArray::CreateDataArray(VTK_ID_TYPE);
      }
      controller->Broadcast(&totalSeeds, 1, 0);
      this->setGlobalElementToCompute(totalSeeds);

      if(myRank != 0) {
        globalSeedsId->SetNumberOfComponents(1);
        globalSeedsId->SetNumberOfTuples(totalSeeds);
      }

      controller->Broadcast(globalSeedsId, 0);

      long int *globalDomainId
        = static_cast<long int *>(ttkUtils::GetVoidPointer(
          domain->GetPointData()->GetArray("GlobalPointIds")));

      int count;
      bool found;
      bool isToCompute;
      for(int i = 0; i < totalSeeds; i++) {
        count = 0;
        found = false;
        isToCompute = false;
        while(count < numberOfPointsInDomain and !found) {
          found = (globalSeedsId->GetTuple1(i) == globalDomainId[count]);
          count++;
        }
        if(found && (pointGhostArray[count - 1] != ttk::type::DUPLICATEPOINT)) {
          vtkInputIdentifiers->InsertNextTuple1(count - 1);
        }
      }

      numberOfPointsInSeeds = vtkInputIdentifiers->GetNumberOfTuples();
    } else {
      this->setGlobalElementToCompute(totalSeeds);
      printMsg("isDistributed!");
      std::vector<SimplexId> idSpareStorage{};
      SimplexId *inputIdentifierGlobalId;
      inputIdentifierGlobalId = this->GetIdentifierArrayPtr(
        ForceInputVertexScalarField, 2, ttk::VertexScalarFieldName, seeds,
        idSpareStorage);
      int localId = 0;
      for(int i = 0; i < numberOfPointsInSeeds; i++) {
        localId = this->getLocalIdFromGlobalId(inputIdentifierGlobalId[i]);
        vtkInputIdentifiers->InsertNextTuple1(localId);
      }
      inputIdentifiers
        = static_cast<SimplexId *>(vtkInputIdentifiers->GetVoidPointer(0));
    }
  } else {
    printMsg("Da0");
    this->setGlobalElementToCompute(numberOfPointsInSeeds);
    std::vector<SimplexId> idSpareStorage{};
    SimplexId *inputIdentifierGlobalId;
    printMsg("Da1");
    inputIdentifierGlobalId = this->GetIdentifierArrayPtr(
      ForceInputVertexScalarField, 2, ttk::VertexScalarFieldName, seeds,
      idSpareStorage);
    printMsg("Da2");
    int localId = 0;
    for(int i = 0; i < numberOfPointsInSeeds; i++) {
      localId = this->getLocalIdFromGlobalId(inputIdentifierGlobalId[i]);
      vtkInputIdentifiers->InsertNextTuple1(localId);
      printMsg("Bla" + std::to_string(i));
    }
    printMsg("Da3");
    inputIdentifiers
      = static_cast<SimplexId *>(vtkInputIdentifiers->GetVoidPointer(0));
  }
  printMsg("Da4");
#else
  std::vector<SimplexId> idSpareStorage{};
  inputIdentifiers = this->GetIdentifierArrayPtr(ForceInputVertexScalarField, 2,
                                                 ttk::VertexScalarFieldName,
                                                 seeds, idSpareStorage);
#endif

#ifndef TTK_ENABLE_KAMIKAZE
  // triangulation problem
  if(!triangulation) {
    this->printErr("wrong triangulation.");
    return -1;
  }
  // field problem
  if(!inputScalars) {
    this->printErr("wrong scalar field.");
    return -1;
  }
  // field problem
  if(inputOffsets->GetDataType() != VTK_INT
     and inputOffsets->GetDataType() != VTK_ID_TYPE) {
    this->printErr("input offset field type not supported.");
    return -1;
  }
  // field problem
  if(!inputIdentifiers) {
    this->printErr("wrong identifiers.");
    return -1;
  }
  // no points.
  if(numberOfPointsInDomain <= 0) {
    this->printErr("domain has no points.");
    return -1;
  }
  // no points.
  // if(totalSeeds <= 0) {
  //   this->printErr("seeds have no points.");
  //   return -1;
  // }
#endif
  LinkedList<vector<SimplexId> *, TABULAR_SIZE> trajectories;
  LinkedList<vector<double> *, TABULAR_SIZE> distancesFromSeed;
  LinkedList<int, TABULAR_SIZE> seedIdentifiers;

  this->setVertexNumber(numberOfPointsInDomain);
  this->setSeedNumber(numberOfPointsInSeeds);
  this->setDirection(Direction);
  this->setInputScalarField(inputScalars->GetVoidPointer(0));
  this->setInputOffsets(
    static_cast<SimplexId *>(inputOffsets->GetVoidPointer(0)));

  this->setVertexIdentifierScalarField(inputIdentifiers);
  this->setOutputTrajectories(&trajectories);
  this->setOutputDistancesFromSeed(&distancesFromSeed);
  this->setOutputSeedIdentifiers(&seedIdentifiers);

  this->preconditionTriangulation(triangulation);
  printMsg("Beginning computation");
  int status = 0;
  ttkVtkTemplateMacro(inputScalars->GetDataType(), triangulation->getType(),
                      (status = this->execute<VTK_TT, TTK_TT>(
                         static_cast<TTK_TT *>(triangulation->getData()))));
#ifndef TTK_ENABLE_KAMIKAZE
  // something wrong in baseCode
  if(status) {
    std::stringstream msg;
    msg << "IntegralLines.execute() error code : " << status;
    this->printErr(msg.str());
    return -1;
  }
#endif

  // make the vtk trajectories
  getTrajectories(domain, triangulation, trajectories, distancesFromSeed,
                  seedIdentifiers, output);

  return (int)(status == 0);
}
