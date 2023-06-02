#include "BaseClass.h"
#include "DataTypes.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include <string>
#include <ttkPeriodicGhostsGeneration.h>

#include <vtkCellData.h>
#include <vtkCharArray.h>
#include <vtkImageAppend.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkStructuredPoints.h>

#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <ttkMacros.h>
#include <ttkUtils.h>

// A VTK macro that enables the instantiation of this class via ::New()
// You do not have to modify this
vtkStandardNewMacro(ttkPeriodicGhostsGeneration);

ttkPeriodicGhostsGeneration::ttkPeriodicGhostsGeneration() {
  this->setDebugMsgPrefix("PeriodicGhostsGeneration");
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
}

int ttkPeriodicGhostsGeneration::FillInputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
    return 1;
  }
  return 0;
}

int ttkPeriodicGhostsGeneration::RequestUpdateExtent(
  vtkInformation *ttkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *ttkNotUsed(outputVector)) {
  vtkImageData *image = vtkImageData::GetData(inputVector[0]);
  this->ComputeOutputExtent(image);
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  inInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
              outExtent_[0] + 1, outExtent_[1] - 1, outExtent_[2] + 1,
              outExtent_[3] - 1, outExtent_[4] + 1, outExtent_[5] - 1);
  return 1;
}

int ttkPeriodicGhostsGeneration::RequestInformation(
  vtkInformation *ttkNotUsed(request),
  vtkInformationVector **inputVectors,
  vtkInformationVector *outputVector) {
  vtkImageData *image = vtkImageData::GetData(inputVectors[0]);
  this->ComputeOutputExtent(image);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  outInfo->Set(
    vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), outExtent_.data(), 6);
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT());
  return 1;
}

int ttkPeriodicGhostsGeneration::FillOutputPortInformation(
  int port, vtkInformation *info) {
  if(port == 0) {
    info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
    return 1;
  }
  return 0;
}

int ttkPeriodicGhostsGeneration::ComputeOutputExtent(vtkDataSet *input) {
  if(!isOutputExtentComputed_) {
    vtkImageData *imageIn;
    if(input->IsA("vtkImageData")) {
      imageIn = vtkImageData::SafeDownCast(input);
    } else {
      printErr("Invalid data input type for periodicTriangulation computation");
      return -1;
    }

    std::array<double, 6> tempGlobalBounds{};
    double bounds[6];
    // Initialize neighbors vector
    imageIn->GetBounds(bounds);
    ttk::preconditionNeighborsUsingBoundingBox(bounds, neighbors_);
    // Reorganize bounds to only execute Allreduce twice
    std::array<double, 6> tempBounds = {
      bounds[0], bounds[2], bounds[4], bounds[1], bounds[3], bounds[5],
    };

    // Compute and send to all processes the lower bounds of the data set
    MPI_Allreduce(tempBounds.data(), tempGlobalBounds.data(), 3, MPI_DOUBLE,
                  MPI_MIN, ttk::MPIcomm_);
    // Compute and send to all processes the higher bounds of the data set
    MPI_Allreduce(&tempBounds[3], &tempGlobalBounds[3], 3, MPI_DOUBLE, MPI_MAX,
                  ttk::MPIcomm_);

    // re-order tempGlobalBounds
    globalBounds_
      = {tempGlobalBounds[0], tempGlobalBounds[3], tempGlobalBounds[1],
         tempGlobalBounds[4], tempGlobalBounds[2], tempGlobalBounds[5]};

    double spacing[3];
    imageIn->GetSpacing(spacing);
    boundsWithoutGhosts_
      = {bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]};
    for(int i = 0; i < 3; i++) {
      if(bounds[2 * i] != globalBounds_[2 * i]) {
        boundsWithoutGhosts_[2 * i] = bounds[2 * i] + spacing[0];
      } else {
      }
      if(bounds[2 * i + 1] != globalBounds_[2 * i + 1]) {
        boundsWithoutGhosts_[2 * i + 1] = bounds[2 * i + 1] - spacing[0];
      } else {
      }
    }
    for(int i = 0; i < 2; i++) {
      if(globalBounds_[i] == boundsWithoutGhosts_[i]) {
        localGlobalBounds_[i].isBound = 1;
        localGlobalBounds_[i].x = boundsWithoutGhosts_[i];
        localGlobalBounds_[i].y
          = (boundsWithoutGhosts_[2] + boundsWithoutGhosts_[3]) / 2;
        localGlobalBounds_[i].z
          = (boundsWithoutGhosts_[4] + boundsWithoutGhosts_[5]) / 2;
      }
    }

    for(int i = 0; i < 2; i++) {
      if(globalBounds_[2 + i] == boundsWithoutGhosts_[2 + i]) {
        localGlobalBounds_[2 + i].isBound = 1;
        localGlobalBounds_[2 + i].x
          = (boundsWithoutGhosts_[0] + boundsWithoutGhosts_[1]) / 2;
        localGlobalBounds_[2 + i].y = boundsWithoutGhosts_[2 + i];
        localGlobalBounds_[2 + i].z
          = (boundsWithoutGhosts_[4] + boundsWithoutGhosts_[5]) / 2;
      }
    }

    for(int i = 0; i < 2; i++) {
      if(globalBounds_[4 + i] == boundsWithoutGhosts_[4 + i]) {
        localGlobalBounds_[4 + i].isBound = 1;
        localGlobalBounds_[4 + i].x
          = (boundsWithoutGhosts_[0] + boundsWithoutGhosts_[1]) / 2;
        localGlobalBounds_[4 + i].y
          = (boundsWithoutGhosts_[2] + boundsWithoutGhosts_[3]) / 2;
        localGlobalBounds_[4 + i].z = boundsWithoutGhosts_[4 + i];
      }
    }

    for(int i = 0; i < 3; i++) {
      if(!(localGlobalBounds_[2 * i].isBound == 1
           && localGlobalBounds_[2 * i + 1].isBound == 1)) {
        outExtent_[2 * i]
          = static_cast<int>(round(globalBounds_[2 * i] / spacing[i])) - 1;
        outExtent_[2 * i + 1]
          = static_cast<int>(round(globalBounds_[2 * i + 1] / spacing[i])) + 1;
      } else {
        outExtent_[2 * i]
          = static_cast<int>(round(globalBounds_[2 * i] / spacing[i]));
        outExtent_[2 * i + 1]
          = static_cast<int>(round(globalBounds_[2 * i + 1] / spacing[i]));
      }
    }
    isOutputExtentComputed_ = true;
  }
  return 1;
};

template <int matchesSize, int metaDataSize>
int ttkPeriodicGhostsGeneration::MarshalAndSendRecv(
  vtkImageData *imageIn,
  std::vector<std::vector<vtkSmartPointer<vtkCharArray>>> &charArrayBoundaries,
  std::vector<std::vector<std::array<ttk::SimplexId, metaDataSize>>>
    &charArrayBoundariesMetaData,
  std::vector<std::array<ttk::SimplexId, matchesSize>> &matches,
  std::vector<vtkSmartPointer<vtkCharArray>> &charArrayBoundariesReceived,
  std::vector<std::array<ttk::SimplexId, metaDataSize>>
    &charArrayBoundariesMetaDataReceived,
  int dim) {
  int *default_VOI = imageIn->GetExtent();
  std::array<ttk::SimplexId, 6> VOI;
  for(int i = 0; i < static_cast<int>(matches.size()); i++) {
    VOI = {default_VOI[0], default_VOI[1], default_VOI[2],
           default_VOI[3], default_VOI[4], default_VOI[5]};
    for(int k = 1; k <= dim; k++) {
      if(matches[i][k] % 2 == 0) {
        VOI[matches[i][k] + 1] = VOI[matches[i][k]];
      } else {
        VOI[matches[i][k] - 1] = VOI[matches[i][k]];
      }
    }
    vtkSmartPointer<vtkExtractVOI> extractVOI
      = vtkSmartPointer<vtkExtractVOI>::New();
    extractVOI->SetInputData(imageIn);
    extractVOI->SetVOI(VOI[0], VOI[1], VOI[2], VOI[3], VOI[4], VOI[5]);
    extractVOI->Update();
    vtkSmartPointer<vtkImageData> extracted = extractVOI->GetOutput();
    vtkSmartPointer<vtkCharArray> buffer = vtkSmartPointer<vtkCharArray>::New();
    if(vtkCommunicator::MarshalDataObject(extracted, buffer) == 0) {
      printErr("Marshalling failed!");
    };
    charArrayBoundaries[matches[i][0]].emplace_back(buffer);
    std::array<ttk::SimplexId, metaDataSize> metaData;
    for(int j = 0; j < metaDataSize; j++) {
      metaData.at(j) = matches[i][j + 1];
    }
    charArrayBoundariesMetaData[matches[i][0]].emplace_back(metaData);
  }

  ttk::SimplexId recv_size;
  ttk::SimplexId send_size;
  std::array<ttk::SimplexId, metaDataSize> sendMetadata;
  std::array<ttk::SimplexId, metaDataSize> recvMetaData;
  for(int i = 0; i < ttk::MPIsize_; i++) {
    for(int j = 0; j < static_cast<int>(charArrayBoundaries[i].size()); j++) {
      send_size = charArrayBoundaries[i][j]->GetNumberOfTuples();
      MPI_Sendrecv(&send_size, 1, ttk::getMPIType(send_size), i, 0, &recv_size,
                   1, ttk::getMPIType(recv_size), i, 0, ttk::MPIcomm_,
                   MPI_STATUS_IGNORE);
      vtkSmartPointer<vtkCharArray> buffer
        = vtkSmartPointer<vtkCharArray>::New();
      buffer->SetNumberOfTuples(recv_size);
      buffer->SetNumberOfComponents(1);
      char *sendPointer = ttkUtils::GetPointer<char>(charArrayBoundaries[i][j]);
      char *recvPointer = ttkUtils::GetPointer<char>(buffer);
      MPI_Sendrecv(sendPointer, send_size, MPI_CHAR, i, 0, recvPointer,
                   recv_size, MPI_CHAR, i, 0, ttk::MPIcomm_, MPI_STATUS_IGNORE);
      charArrayBoundariesReceived.emplace_back(buffer);
      sendMetadata = charArrayBoundariesMetaData[i][j];
      MPI_Sendrecv(sendMetadata.data(), metaDataSize,
                   ttk::getMPIType(recv_size), i, 0, recvMetaData.data(),
                   metaDataSize, ttk::getMPIType(recv_size), i, 0,
                   ttk::MPIcomm_, MPI_STATUS_IGNORE);
      charArrayBoundariesMetaDataReceived.emplace_back(recvMetaData);
    }
  }
  return 1;
}

template <typename boundaryType>
int ttkPeriodicGhostsGeneration::UnMarshalAndMerge(
  std::vector<boundaryType> &metaDataReceived,
  std::vector<vtkSmartPointer<vtkCharArray>> &boundariesReceived,
  boundaryType direction,
  int mergeDirection,
  vtkImageData *mergedImage) {
  auto it
    = std::find(metaDataReceived.begin(), metaDataReceived.end(), direction);
  if(it != metaDataReceived.end()) {
    vtkNew<vtkStructuredPoints> id;
    vtkNew<vtkImageData> aux;
    if(vtkCommunicator::UnMarshalDataObject(
         boundariesReceived[std::distance(metaDataReceived.begin(), it)], id)
       == 0) {
      printErr("UnMarshaling failed!");
      return 0;
    };
    this->MergeImageAppendAndSlice(mergedImage, id, aux, mergeDirection);
    mergedImage->DeepCopy(aux);
  }
  return 1;
}

template <typename boundaryType>
int ttkPeriodicGhostsGeneration::UnMarshalAndCopy(
  std::vector<boundaryType> &metaDataReceived,
  std::vector<vtkSmartPointer<vtkCharArray>> &boundariesReceived,
  boundaryType direction,
  vtkImageData *mergedImage) {
  auto it
    = std::find(metaDataReceived.begin(), metaDataReceived.end(), direction);
  if(it != metaDataReceived.end()) {
    vtkNew<vtkStructuredPoints> id;
    if(vtkCommunicator::UnMarshalDataObject(
         boundariesReceived[std::distance(metaDataReceived.begin(), it)], id)
       == 0) {
      printErr("UnMarshaling failed!");
      return 0;
    };
    mergedImage->DeepCopy(id);
  }
  return 1;
}

int ttkPeriodicGhostsGeneration::MergeDataArrays(
  vtkDataArray *imageArray,
  vtkDataArray *sliceArray,
  vtkSmartPointer<vtkDataArray> &currentArray,
  int direction,
  int dims[3],
  unsigned char ghostValue,
  ttk::SimplexId numberOfSimplices,
  ttk::SimplexId numberOfTuples) {
  std::string arrayName(imageArray->GetName());

#ifndef TTK_ENABLE_KAMIKAZE
  if(!sliceArray) {
    printErr("Array " + arrayName
             + " is not present in the Data of the second vtkImageData");
    return 0;
  }
#endif
  currentArray->SetNumberOfComponents(1);
  currentArray->SetNumberOfTuples(numberOfSimplices);
  currentArray->SetName(arrayName.c_str());
  if(std::strcmp(currentArray->GetName(), "vtkGhostType") == 0) {
    sliceArray->SetNumberOfTuples(numberOfTuples);
    sliceArray->Fill(ghostValue);
  }
  int sliceCounter = 0;
  int imageCounter = 0;
  int counter = 0;
  std::function<bool(int, int, int, int[3])> addSlice;
  switch(direction) {
    case 0:
      addSlice = [](int x, int ttkNotUsed(y), int ttkNotUsed(z),
                    int ttkNotUsed(dimensions)[3]) { return x == 0; };
      break;
    case 1:
      addSlice = [](int x, int ttkNotUsed(y), int ttkNotUsed(z),
                    int dimensions[3]) { return x == dimensions[0] - 1; };
      break;
    case 2:
      addSlice = [](int ttkNotUsed(x), int y, int ttkNotUsed(z),
                    int ttkNotUsed(dimensions)[3]) { return y == 0; };
      break;
    case 3:
      addSlice = [](int ttkNotUsed(x), int y, int ttkNotUsed(z),
                    int dimensions[3]) { return y == dimensions[1] - 1; };
      break;
    case 4:
      addSlice = [](int ttkNotUsed(x), int ttkNotUsed(y), int z,
                    int ttkNotUsed(dimensions)[3]) { return z == 0; };
      break;
    case 5:
      addSlice = [](int ttkNotUsed(x), int ttkNotUsed(y), int z,
                    int dimensions[3]) { return z == dimensions[2] - 1; };
      break;
  }

  for(int z = 0; z < dims[2]; z++) {
    for(int y = 0; y < dims[1]; y++) {
      for(int x = 0; x < dims[0]; x++) {
        if(addSlice(x, y, z, dims)) {
          currentArray->SetTuple1(counter, sliceArray->GetTuple1(sliceCounter));
          sliceCounter++;
        } else {
          currentArray->SetTuple1(counter, imageArray->GetTuple1(imageCounter));
          imageCounter++;
        }
        counter++;
      }
    }
  }
  return 1;
}

int ttkPeriodicGhostsGeneration::MergeImageAppendAndSlice(
  vtkImageData *image,
  vtkImageData *slice,
  vtkImageData *mergedImage,
  int direction) {
  // TODO: handle GlobalPointIds
#ifndef TTK_ENABLE_KAMIKAZE
  if(image->GetPointData()->GetNumberOfArrays()
     != slice->GetPointData()->GetNumberOfArrays()) {
    printErr("The two vtkImageData objects to merge don't have the same number "
             "of arrays for their Point Data");
    return 0;
  }

  if(image->GetCellData()->GetNumberOfArrays()
     != slice->GetCellData()->GetNumberOfArrays()) {
    printErr("The two vtkImageData objects to merge don't have the same number "
             "of arrays for their Cell Data");
    return 0;
  }
#endif
  mergedImage->SetSpacing(image->GetSpacing());
  mergedImage->Initialize();
  int extentImage[6];
  image->GetExtent(extentImage);
  if(direction % 2 == 0) {
    extentImage[direction] -= 1;
  } else {
    extentImage[direction] += 1;
  }
  mergedImage->SetExtent(extentImage);
  int dims[3];
  mergedImage->GetDimensions(dims);
  int cellDims[3] = {dims[0] - 1, dims[1] - 1, dims[2] - 1};
  int numberOfPoints = mergedImage->GetNumberOfPoints();
  for(int array = 0; array < image->GetPointData()->GetNumberOfArrays();
      array++) {
    vtkDataArray *imageArray = image->GetPointData()->GetArray(array);
    vtkDataArray *sliceArray
      = slice->GetPointData()->GetArray(imageArray->GetName());
    vtkSmartPointer<vtkDataArray> currentArray
      = vtkSmartPointer<vtkDataArray>::Take(imageArray->NewInstance());
    this->MergeDataArrays(imageArray, sliceArray, currentArray, direction, dims,
                          vtkDataSetAttributes::DUPLICATEPOINT, numberOfPoints,
                          slice->GetNumberOfPoints());
    mergedImage->GetPointData()->AddArray(currentArray);
  }
  int numberOfCells = mergedImage->GetNumberOfCells();

  for(int array = 0; array < image->GetCellData()->GetNumberOfArrays();
      array++) {
    vtkDataArray *imageArray = image->GetCellData()->GetArray(array);
    std::string arrayName(imageArray->GetName());
    vtkDataArray *sliceArray
      = slice->GetCellData()->GetArray(arrayName.c_str());
    if(std::strcmp(arrayName.c_str(), "Cell Type") == 0) {
      break;
    }
    vtkSmartPointer<vtkDataArray> currentArray
      = vtkSmartPointer<vtkDataArray>::Take(imageArray->NewInstance());
    if(direction == 0 || direction == 2 || direction == 4) {
      this->MergeDataArrays(imageArray, sliceArray, currentArray, direction,
                            cellDims, 0, numberOfCells,
                            slice->GetNumberOfCells());
    } else {
      this->MergeDataArrays(imageArray, sliceArray, currentArray, direction,
                            cellDims, vtkDataSetAttributes::EXTERIORCELL,
                            numberOfCells, slice->GetNumberOfCells());
    }

    mergedImage->GetCellData()->AddArray(currentArray);
  }

  vtkNew<vtkCharArray> cellTypes;
  cellTypes->SetName("Cell Type");
  cellTypes->SetNumberOfTuples(numberOfCells);
  cellTypes->Fill(image->GetCellType(0));
  mergedImage->GetCellData()->AddArray(cellTypes);

  return 1;
};

int ttkPeriodicGhostsGeneration::MPIPeriodicGhostPipelinePreconditioning(
  vtkImageData *imageIn, vtkImageData *imageOut) {

  if(!ttk::isRunningWithMPI()) {
    return 0;
  }
  auto other = [](ttk::SimplexId i) {
    if(i % 2 == 1) {
      return i - 1;
    }
    return i + 1;
  };

  int *dims = imageIn->GetDimensions();
  int dimensionality = imageIn->GetDataDimension();
  // Case dimensionality == 3
  int firstDim = 4;
  int secondDim = 2;

  if(dimensionality == 1) {
    firstDim = 0;
    if(dims[0] <= 1 && dims[1] <= 1) {
      firstDim = 4;
    } else {
      if(dims[0] <= 1 && dims[2] <= 1) {
        firstDim = 2;
      }
    }
  }
  if(dimensionality == 2) {
    if(dims[0] <= 1) {
      firstDim = 4;
      secondDim = 2;
    }
    if(dims[1] <= 1) {
      firstDim = 4;
      secondDim = 0;
    }
    if(dims[2] <= 1) {
      firstDim = 2;
      secondDim = 0;
    }
  }

  this->ComputeOutputExtent(imageIn);

  MPI_Datatype partialGlobalBoundMPI;
  std::vector<periodicGhosts::partialGlobalBound> allLocalGlobalBounds(
    ttk::MPIsize_ * 6, periodicGhosts::partialGlobalBound{});
  MPI_Datatype types[]
    = {MPI_UNSIGNED_CHAR, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
  int lengths[] = {1, 1, 1, 1};
  const long int mpi_offsets[]
    = {offsetof(periodicGhosts::partialGlobalBound, isBound),
       offsetof(periodicGhosts::partialGlobalBound, x),
       offsetof(periodicGhosts::partialGlobalBound, y),
       offsetof(periodicGhosts::partialGlobalBound, z)};
  MPI_Type_create_struct(
    4, lengths, mpi_offsets, types, &partialGlobalBoundMPI);
  MPI_Type_commit(&partialGlobalBoundMPI);

  MPI_Allgather(localGlobalBounds_.data(), 6, partialGlobalBoundMPI,
                allLocalGlobalBounds.data(), 6, partialGlobalBoundMPI,
                ttk::MPIcomm_);

  std::vector<std::array<ttk::SimplexId, 3>> matches;
  for(int i = 0; i < ttk::MPIsize_; i++) {
    if(i != ttk::MPIrank_) {
      for(int j = 0; j < 6; j++) {
        bool isIn = false;
        if(!(localGlobalBounds_[other(j)].isBound != 0
             && localGlobalBounds_[j].isBound != 0)) {
          if(localGlobalBounds_[other(j)].isBound != 0
             && allLocalGlobalBounds[i * 6 + j].isBound != 0) {
            if(0 <= j && j <= 1) {
              isIn
                = (boundsWithoutGhosts_[2] <= allLocalGlobalBounds[i * 6 + j].y
                   && boundsWithoutGhosts_[3]
                        >= allLocalGlobalBounds[i * 6 + j].y)
                  && (boundsWithoutGhosts_[4]
                        <= allLocalGlobalBounds[i * 6 + j].z
                      && boundsWithoutGhosts_[5]
                           >= allLocalGlobalBounds[i * 6 + j].z);
            }
            if(2 <= j && j <= 3) {
              isIn
                = (boundsWithoutGhosts_[0] <= allLocalGlobalBounds[i * 6 + j].x
                   && boundsWithoutGhosts_[1]
                        >= allLocalGlobalBounds[i * 6 + j].x)
                  && (boundsWithoutGhosts_[4]
                        <= allLocalGlobalBounds[i * 6 + j].z
                      && boundsWithoutGhosts_[5]
                           >= allLocalGlobalBounds[i * 6 + j].z);
            }
            if(4 <= j && j <= 5) {
              isIn
                = (boundsWithoutGhosts_[0] <= allLocalGlobalBounds[i * 6 + j].x
                   && boundsWithoutGhosts_[1]
                        >= allLocalGlobalBounds[i * 6 + j].x)
                  && (boundsWithoutGhosts_[2]
                        <= allLocalGlobalBounds[i * 6 + j].y
                      && boundsWithoutGhosts_[3]
                           >= allLocalGlobalBounds[i * 6 + j].y);
            }
            if(isIn) {
              matches.emplace_back(
                std::array<ttk::SimplexId, 3>{i, other(j), j});
              if(std::find(neighbors_.begin(), neighbors_.end(), i)
                 == neighbors_.end()) {
                neighbors_.push_back(i);
              }
            }
          }
        }
      }
    }
  }

  std::vector<std::array<ttk::SimplexId, 2>> local2DBounds;
  std::vector<std::array<ttk::SimplexId, 3>> matches_2D;
  if(dimensionality >= 2) {
    for(int i = 0; i < 4; i++) {
      for(int j = i + 1; j < 6; j++) {
        if((abs(i - j) == 1 && i % 2 == 1) || abs(i - j) >= 2) {
          if((localGlobalBounds_[i].isBound != 0
              && localGlobalBounds_[j].isBound != 0)
             && !(localGlobalBounds_[other(i)].isBound != 0
                  && localGlobalBounds_[other(j)].isBound != 0)) {
            local2DBounds.emplace_back(std::array<ttk::SimplexId, 2>{i, j});
          }
        }
      }
    }

    for(int i = 0; i < static_cast<int>(local2DBounds.size()); i++) {
      for(int j = 0; j < ttk::MPIsize_; j++) {
        if(j != ttk::MPIrank_) {
          bool isIn = false;
          if((allLocalGlobalBounds[j * 6 + other(local2DBounds[i][0])].isBound
              != 0)
             && (allLocalGlobalBounds[j * 6 + other(local2DBounds[i][1])]
                   .isBound
                 != 0)) {
            ttk::SimplexId dirs[2]
              = {other(local2DBounds[i][0]), other(local2DBounds[i][1])};
            std::sort(dirs, dirs + 2);
            if((dirs[0] < 2 && dirs[1] >= 2 && dirs[1] < 4)) {
              isIn = (boundsWithoutGhosts_[4]
                        <= allLocalGlobalBounds[j * 6 + dirs[0]].z
                      && boundsWithoutGhosts_[5]
                           >= allLocalGlobalBounds[j * 6 + dirs[0]].z);
            }
            if((dirs[0] < 2 && dirs[1] >= 4)) {
              isIn = (boundsWithoutGhosts_[2]
                        <= allLocalGlobalBounds[j * 6 + dirs[0]].y
                      && boundsWithoutGhosts_[3]
                           >= allLocalGlobalBounds[j * 6 + dirs[0]].y);
            }
            if((dirs[0] >= 2 && dirs[0] < 4 && dirs[1] >= 4)) {
              isIn = (boundsWithoutGhosts_[0]
                        <= allLocalGlobalBounds[j * 6 + dirs[0]].x
                      && boundsWithoutGhosts_[1]
                           >= allLocalGlobalBounds[j * 6 + dirs[0]].x);
            }
            if(isIn) {
              matches_2D.emplace_back(std::array<ttk::SimplexId, 3>{
                j, local2DBounds[i][0], local2DBounds[i][1]});
              if(std::find(neighbors_.begin(), neighbors_.end(), j)
                 == neighbors_.end()) {
                neighbors_.push_back(j);
              }
            }
          }
        }
      }
    }
  }
  std::vector<std::array<ttk::SimplexId, 3>> local3DBounds;
  std::vector<std::array<ttk::SimplexId, 4>> matches_3D;
  if(dimensionality == 3) {
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
        for(int k = 0; k < 2; k++) {
          if(localGlobalBounds_[i].isBound != 0
             && localGlobalBounds_[2 + j].isBound != 0
             && localGlobalBounds_[4 + k].isBound != 0) {
            local3DBounds.emplace_back(
              std::array<ttk::SimplexId, 3>{i, 2 + j, 4 + k});
          }
        }
      }
    }
    for(int i = 0; i < static_cast<int>(local3DBounds.size()); i++) {
      for(int j = 0; j < ttk::MPIsize_; j++) {
        if(j != ttk::MPIrank_) {
          if((allLocalGlobalBounds[j * 6 + other(local3DBounds[i][0])].isBound
              != 0)
             && (allLocalGlobalBounds[j * 6 + other(local3DBounds[i][1])]
                   .isBound
                 != 0)
             && (allLocalGlobalBounds[j * 6 + other(local3DBounds[i][2])]
                   .isBound
                 != 0)) {
            matches_3D.emplace_back(std::array<ttk::SimplexId, 4>{
              j, local3DBounds[i][0], local3DBounds[i][1],
              local3DBounds[i][2]});
            if(std::find(neighbors_.begin(), neighbors_.end(), j)
               == neighbors_.end()) {
              neighbors_.push_back(j);
            }
          }
        }
      }
    }
  }
  // Now, extract ImageData for 1D boundaries
  std::vector<std::vector<vtkSmartPointer<vtkCharArray>>> charArrayBoundaries(
    ttk::MPIsize_);
  std::vector<std::vector<std::array<ttk::SimplexId, 1>>>
    charArrayBoundariesMetaData(ttk::MPIsize_);
  std::vector<vtkSmartPointer<vtkCharArray>> charArrayBoundariesReceived;
  std::vector<std::array<ttk::SimplexId, 1>>
    charArrayBoundariesMetaDataReceived;
  if(this->MarshalAndSendRecv<3, 1>(
       imageIn, charArrayBoundaries, charArrayBoundariesMetaData, matches,
       charArrayBoundariesReceived, charArrayBoundariesMetaDataReceived, 1)
     == 0) {
    return 0;
  }

  // Extract 2D boundaries
  std::vector<std::vector<vtkSmartPointer<vtkCharArray>>> charArray2DBoundaries(
    ttk::MPIsize_);
  std::vector<std::vector<std::array<ttk::SimplexId, 2>>>
    charArray2DBoundariesMetaData(ttk::MPIsize_);
  std::vector<vtkSmartPointer<vtkCharArray>> charArray2DBoundariesReceived;
  std::vector<std::array<ttk::SimplexId, 2>>
    charArray2DBoundariesMetaDataReceived;
  if(dimensionality >= 2) {
    if(this->MarshalAndSendRecv<3, 2>(imageIn, charArray2DBoundaries,
                                      charArray2DBoundariesMetaData, matches_2D,
                                      charArray2DBoundariesReceived,
                                      charArray2DBoundariesMetaDataReceived, 2)
       == 0) {
      return 0;
    }
  }
  // Now, same for 3D boundaries
  std::vector<std::vector<vtkSmartPointer<vtkCharArray>>> charArray3DBoundaries(
    ttk::MPIsize_);
  std::vector<std::vector<std::array<ttk::SimplexId, 3>>>
    charArray3DBoundariesMetaData(ttk::MPIsize_);
  std::vector<vtkSmartPointer<vtkCharArray>> charArray3DBoundariesReceived;
  std::vector<std::array<ttk::SimplexId, 3>>
    charArray3DBoundariesMetaDataReceived;
  if(dimensionality == 3) {
    if(this->MarshalAndSendRecv<4, 3>(imageIn, charArray3DBoundaries,
                                      charArray3DBoundariesMetaData, matches_3D,
                                      charArray3DBoundariesReceived,
                                      charArray3DBoundariesMetaDataReceived, 3)
       == 0) {
      return 0;
    }
  }
  imageOut->DeepCopy(imageIn);

  // Merge in the first direction (low and high)
  for(int dir = firstDim; dir < firstDim + 2; dir++) {
    if(this->UnMarshalAndMerge<std::array<ttk::SimplexId, 1>>(
         charArrayBoundariesMetaDataReceived, charArrayBoundariesReceived,
         std::array<ttk::SimplexId, 1>{other(dir)}, dir, imageOut)
       == 0) {
      return 0;
    }
  }
  if(dimensionality >= 2) {
    // Merge in the second direction
    for(int dir = secondDim; dir < secondDim + 2; dir++) {
      vtkSmartPointer<vtkImageData> mergedImage
        = vtkSmartPointer<vtkImageData>::New();
      if(this->UnMarshalAndCopy<std::array<ttk::SimplexId, 1>>(
           charArrayBoundariesMetaDataReceived, charArrayBoundariesReceived,
           std::array<ttk::SimplexId, 1>{other(dir)}, mergedImage)
         == 0) {
        return 0;
      }
      if(mergedImage->GetNumberOfPoints() > 0) {
        for(int dir_2D = firstDim; dir_2D < firstDim + 2; dir_2D++) {
          if(this->UnMarshalAndMerge<std::array<ttk::SimplexId, 2>>(
               charArray2DBoundariesMetaDataReceived,
               charArray2DBoundariesReceived,
               std::array<ttk::SimplexId, 2>{other(dir), other(dir_2D)}, dir_2D,
               mergedImage)
             == 0) {
            return 0;
          }
        }
        vtkSmartPointer<vtkImageData> aux
          = vtkSmartPointer<vtkImageData>::New();
        if(this->MergeImageAppendAndSlice(imageOut, mergedImage, aux, dir)
           == 0) {
          return 0;
        }
        imageOut->DeepCopy(aux);
      }
    }
  }
  if(dimensionality == 3) {
    // Merge in the x direction
    for(int dir = 0; dir < 2; dir++) {
      vtkNew<vtkImageData> mergedImage1;
      if(this->UnMarshalAndCopy<std::array<ttk::SimplexId, 1>>(
           charArrayBoundariesMetaDataReceived, charArrayBoundariesReceived,
           std::array<ttk::SimplexId, 1>{other(dir)}, mergedImage1)
         == 0) {
        return 0;
      }
      if(mergedImage1->GetNumberOfPoints() > 0) {
        for(int dir_2D = firstDim; dir_2D < firstDim + 2; dir_2D++) {
          if(this->UnMarshalAndMerge<std::array<ttk::SimplexId, 2>>(
               charArray2DBoundariesMetaDataReceived,
               charArray2DBoundariesReceived,
               std::array<ttk::SimplexId, 2>{other(dir), other(dir_2D)}, dir_2D,
               mergedImage1)
             == 0) {
            return 0;
          }
        }
        for(int dir_2D = secondDim; dir_2D < secondDim + 2; dir_2D++) {
          vtkNew<vtkImageData> mergedImage2;
          if(this->UnMarshalAndCopy<std::array<ttk::SimplexId, 2>>(
               charArray2DBoundariesMetaDataReceived,
               charArray2DBoundariesReceived,
               std::array<ttk::SimplexId, 2>{other(dir), other(dir_2D)},
               mergedImage2)
             == 0) {
            return 0;
          }
          if(mergedImage2->GetNumberOfPoints() > 0) {
            for(int dir_3D = firstDim; dir_3D < firstDim + 2; dir_3D++) {
              if(this->UnMarshalAndMerge<std::array<ttk::SimplexId, 3>>(
                   charArray3DBoundariesMetaDataReceived,
                   charArray3DBoundariesReceived,
                   std::array<ttk::SimplexId, 3>{
                     other(dir), other(dir_2D), other(dir_3D)},
                   dir_3D, mergedImage2)
                 == 0) {
                return 0;
              }
            }
            vtkNew<vtkImageData> aux;
            if(this->MergeImageAppendAndSlice(
                 mergedImage1, mergedImage2, aux, dir_2D)
               == 0) {
              return 0;
            }
            mergedImage1->DeepCopy(aux);
          }
        }
        if(mergedImage1->GetNumberOfPoints() > 0) {
          vtkNew<vtkImageData> aux;
          if(this->MergeImageAppendAndSlice(imageOut, mergedImage1, aux, dir)
             == 0) {
            return 0;
          }
          imageOut->DeepCopy(aux);
        }
      }
    }
  }
  return 1;
};

int ttkPeriodicGhostsGeneration::RequestData(
  vtkInformation *ttkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector) {

  vtkImageData *imageIn = vtkImageData::GetData(inputVector[0]);
  vtkImageData *imageOut = vtkImageData::GetData(outputVector);

  this->MPIPeriodicGhostPipelinePreconditioning(imageIn, imageOut);

  // return success
  return 1;
};