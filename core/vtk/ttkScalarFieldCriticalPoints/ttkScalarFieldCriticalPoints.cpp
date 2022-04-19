#include <ttkScalarFieldCriticalPoints.h>

#include <vtkInformation.h>

#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkInformationVector.h>
#include <vtkIntArray.h>
#include <vtkMPIController.h>
#include <vtkMultiProcessController.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkSignedCharArray.h>
#include <vtkUnsignedCharArray.h>

#include <ttkMacros.h>
#include <ttkUtils.h>
#include <DataSetAttributes.h>

using namespace std;
using namespace ttk;

vtkStandardNewMacro(ttkScalarFieldCriticalPoints);

ttkScalarFieldCriticalPoints::ttkScalarFieldCriticalPoints() {

  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

ttkScalarFieldCriticalPoints::~ttkScalarFieldCriticalPoints() = default;

int ttkScalarFieldCriticalPoints::FillInputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0)
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
  else
    return 0;

  return 1;
}

int ttkScalarFieldCriticalPoints::FillOutputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0)
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
  else
    return 0;

  return 1;
}

int ttkScalarFieldCriticalPoints::RequestData(
  vtkInformation *ttkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector) {

  vtkDataSet *input = vtkDataSet::GetData(inputVector[0]);

  vtkPolyData *output = vtkPolyData::GetData(outputVector, 0);

  ttk::Triangulation *triangulation = ttkAlgorithm::GetTriangulation(input);
  if(!triangulation)
    return 0;

  if(VertexBoundary)
    triangulation->preconditionBoundaryVertices();

  // in the following, the target scalar field of the input is replaced in the
  // variable 'output' with the result of the computation.
  // if your wrapper produces an output of the same type of the input, you
  // should proceed in the same way.

  vtkDataArray *inputScalarField = this->GetInputArrayToProcess(0, inputVector);
  if(!inputScalarField)
    return 0;

  vtkDataArray *offsetField
    = this->GetOrderArray(input, 0, 1, ForceInputOffsetScalarField);

  // setting up the base layer
  this->preconditionTriangulation(triangulation);
  this->setOutput(&criticalPoints_);

  #if TTK_ENABLE_MPI
  Timer t_mpi;
  // Get processes information
  vtkMPIController *controller = vtkMPIController::SafeDownCast(
    vtkMultiProcessController::GetGlobalController());
  int myRank = controller->GetLocalProcessId();
  int numberOfProcesses = controller->GetNumberOfProcesses();
  this->setNumberOfProcesses(numberOfProcesses);
  this->setMyRank(myRank);

  controller->Barrier();
  if(myRank == 0) {
    t_mpi.reStart();
  }
  // Set GlobalIdsArray
  long int *globalPointsId = static_cast<long int *>(ttkUtils::GetVoidPointer(
    input->GetPointData()->GetArray("GlobalPointIds")));
  // vtkDataArray* global = input->GetPointData()->GetArray("GlobalPointIds");
  // std::string s = global->GetClassName();
  //   vtkSmartPointer<vtkIdTypeArray> global_temp
  //   = vtkSmartPointer<vtkIdTypeArray>::New();
  // global_temp->SetNumberOfComponents(global->GetNumberOfComponents());
  // global_temp->SetNumberOfTuples(global->GetNumberOfTuples());
  // if (s == "vtkDoubleArray"){
  //   for (int i= 0; i<global->GetNumberOfTuples(); i++){
  //     global_temp->SetTuple1(i, (long int)global->GetTuple1(i));
  //   }
  //   globalPointsId = static_cast<long int *>(ttkUtils::GetVoidPointer(
  //   global_temp));
  //   printMsg("Da");
  // }
  // else{

  // }

  // printMsg("global id: number of components:
  // "+std::to_string(global->GetNumberOfComponents())); printMsg("global id:
  // number of tuples: "+std::to_string(global->GetNumberOfTuples()));
  // printMsg("global id: element 1 from VTK:
  // "+std::to_string(global->GetTuple1(0))); printMsg("global id: element 1
  // from VTK: "+std::to_string(global_temp->GetTuple1(0))); printMsg("global
  // id: element 1 from C++: "+std::to_string(globalPointsId[0]));
  // printMsg("global id: array type "+std::to_string(global->GetArrayType()));
  // printMsg("global id: class name "+s);
  this->setGlobalIdsArray(globalPointsId);

  // int *processId = static_cast<int *>(
  //   ttkUtils::GetVoidPointer(input->GetCellData()->GetArray("ProcessId")));

  if(numberOfProcesses > 1) {

    // Set pointGhostArray
    unsigned char *pointGhostArray = static_cast<unsigned char *>(
      ttkUtils::GetVoidPointer(input->GetGhostArray(vtkDataObject::POINT)));
    // unsigned char *cellGhostArray = static_cast<unsigned char *>(
    //   ttkUtils::GetVoidPointer(input->GetCellGhostArray()));
    this->setPointGhostArray(pointGhostArray);

    // int vertexNumber = triangulation->getNumberOfVertices();

    //     // Stores whether a vertex is on the boundary (doesn't take ghost
    //     points
    //     // into account)
    // vtkSmartPointer<vtkIntArray> isOnMPIBoundary
    //   = vtkSmartPointer<vtkIntArray>::New();
    // isOnMPIBoundary->SetNumberOfComponents(1);
    // isOnMPIBoundary->SetNumberOfTuples(vertexNumber);
    // isOnMPIBoundary->Fill(0);

    // // Stores which process contains this vertex
    // std::vector<std::vector<int>> vertex2Process(
    // vertexNumber, std::vector<int>(1, myRank));

    // int cellVertexNumber = 0;
    // int v_id;
    // int localProcessId;
    // int counter;
    // int sizeVector;

    // // Construct isOnMPIBoundary and vertex2Process
    // for(int id = 0; id < triangulation->getNumberOfCells(); id++) {
    //   if(cellGhostArray[id] && ttk::type::DUPLICATEPOINT) {
    //       cellVertexNumber = triangulation->getCellVertexNumber(id);
    //       for(int vertex = 0; vertex < cellVertexNumber; vertex++) {
    //       triangulation->getCellVertex(id, vertex, v_id);
    //       if(!(pointGhostArray[v_id] && ttk::type::DUPLICATEPOINT)) {
    //         isOnMPIBoundary->SetTuple1(v_id, 1);
    //         localProcessId = processId[id];
    //         counter = 0;
    //         sizeVector = vertex2Process[v_id].size();
    //         while((vertex2Process[v_id][counter] < localProcessId)
    //             && (sizeVector > counter)) {
    //           counter++;
    //         }
    //         if(sizeVector == counter) {
    //           vertex2Process[v_id].push_back(localProcessId);

    //         } else {
    //           if(localProcessId != vertex2Process[v_id][counter]) {
    //     vertex2Process[v_id].insert(vertex2Process[v_id].begin()+counter,
    //     localProcessId);
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    // this->setVertex2Process(vertex2Process);
    // this->setIsOnMPIBoundary(
    //         static_cast<int *>(ttkUtils::GetVoidPointer(isOnMPIBoundary)));
  }
  controller->Barrier();
  if(myRank == 0) {
    printMsg("Preparation performed using " + std::to_string(numberOfProcesses)
               + " MPI processes lasted :"+ std::to_string(t_mpi.getElapsedTime()));
  }
  controller->Barrier();
  if(myRank == 0) {
	t_mpi.reStart();
  }

#endif
  
  printMsg("Starting computation...");
  printMsg({{"  Scalar Array", inputScalarField->GetName()},
            {"  Offset Array", offsetField ? offsetField->GetName() : "None"}});
  int status = 0;
  ttkTemplateMacro(
    triangulation->getType(),
    (status = this->execute(
       static_cast<SimplexId *>(ttkUtils::GetVoidPointer(offsetField)),
       (TTK_TT *)triangulation->getData())));

  if(status == -1) {
    vtkErrorMacro("Please use Ghost Arrays for parallel computation of critical points");
  }
  if(status < 0)
    return 0;
#if TTK_ENABLE_MPI
  controller->Barrier();

  if(myRank == 0) {
    printMsg("Computation performed using " + std::to_string(numberOfProcesses)
              + " MPI processes lasted :"+ std::to_string(t_mpi.getElapsedTime()));
  }
#endif

  // allocate the output
  vtkNew<vtkSignedCharArray> vertexTypes{};
  vertexTypes->SetNumberOfComponents(1);
  vertexTypes->SetNumberOfTuples(criticalPoints_.size());
  vertexTypes->SetName("CriticalType");

  vtkNew<vtkPoints> pointSet{};
  pointSet->SetNumberOfPoints(criticalPoints_.size());

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(this->threadNumber_)
#endif // TTK_ENABLE_OPENMP
  for(size_t i = 0; i < criticalPoints_.size(); i++) {
    std::array<double, 3> p{};
    input->GetPoint(criticalPoints_[i].first, p.data());
    pointSet->SetPoint(i, p.data());
    vertexTypes->SetTuple1(i, (float)criticalPoints_[i].second);
  }

  ttkUtils::CellVertexFromPoints(output, pointSet);
  output->GetPointData()->AddArray(vertexTypes);

  if(VertexBoundary) {
    vtkNew<vtkSignedCharArray> vertexBoundary{};
    vertexBoundary->SetNumberOfComponents(1);
    vertexBoundary->SetNumberOfTuples(criticalPoints_.size());
    vertexBoundary->SetName("IsOnBoundary");

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(size_t i = 0; i < criticalPoints_.size(); i++) {
      vertexBoundary->SetTuple1(
        i, (signed char)triangulation->isVertexOnBoundary(
             criticalPoints_[i].first));
    }

    output->GetPointData()->AddArray(vertexBoundary);
  } else {
    output->GetPointData()->RemoveArray("IsOnBoundary");
  }
  if(VertexIds) {
    vtkNew<ttkSimplexIdTypeArray> vertexIds{};
    vertexIds->SetNumberOfComponents(1);
    vertexIds->SetNumberOfTuples(criticalPoints_.size());
    vertexIds->SetName(ttk::VertexScalarFieldName);

    for(size_t i = 0; i < criticalPoints_.size(); i++) {
#if TTK_ENABLE_MPI
      vertexIds->SetTuple1(i, this->GlobalIdsArray[criticalPoints_[i].first]);
#else
      vertexIds->SetTuple1(i, criticalPoints_[i].first);
#endif
    }

    output->GetPointData()->AddArray(vertexIds);
  } else {
    output->GetPointData()->RemoveArray(ttk::VertexScalarFieldName);
  }
  if(VertexScalars) {
    for(SimplexId i = 0; i < input->GetPointData()->GetNumberOfArrays(); i++) {

      vtkDataArray *scalarField = input->GetPointData()->GetArray(i);
      vtkSmartPointer<vtkDataArray> scalarArray{scalarField->NewInstance()};

      scalarArray->SetNumberOfComponents(scalarField->GetNumberOfComponents());
      scalarArray->SetNumberOfTuples(criticalPoints_.size());
      scalarArray->SetName(scalarField->GetName());
      std::vector<double> value(scalarField->GetNumberOfComponents());
      for(size_t j = 0; j < criticalPoints_.size(); j++) {
        scalarField->GetTuple(criticalPoints_[j].first, value.data());
        scalarArray->SetTuple(j, value.data());
      }
      output->GetPointData()->AddArray(scalarArray);
    }
  } else {
    for(SimplexId i = 0; i < input->GetPointData()->GetNumberOfArrays(); i++) {
      output->GetPointData()->RemoveArray(
        input->GetPointData()->GetArray(i)->GetName());
    }
  }
  return 1;
}
