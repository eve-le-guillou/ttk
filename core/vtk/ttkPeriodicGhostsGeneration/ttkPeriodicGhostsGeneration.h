/// TODO 4: Provide your information and **update** the documentation (in
/// particular regarding the order convention if input arrays need to be
/// specified with the standard VTK call SetInputArrayToProcess()).
///
/// \ingroup vtk
/// \class ttkPeriodicGhostsGeneration
/// \author Your Name Here <your.email@address.here>
/// \date The Date Here.
///
/// \brief TTK VTK-filter that wraps the ttk::PeriodicGhostsGeneration module.
///
/// This VTK filter uses the ttk::PeriodicGhostsGeneration module to compute an
/// averaging of the data values of an input point data array defined on the
/// input vtkDataSet.
///
/// \param Input vtkDataSet.
/// \param Output vtkDataSet.
///
/// This filter can be used as any other VTK filter (for instance, by using the
/// sequence of calls SetInputData(), Update(), GetOutputDataObject()).
///
/// The input data array needs to be specified via the standard VTK call
/// vtkAlgorithm::SetInputArrayToProcess() with the following parameters:
/// \param idx 0 (FIXED: the first array the algorithm requires)
/// \param port 0 (FIXED: first port)
/// \param connection 0 (FIXED: first connection)
/// \param fieldAssociation 0 (FIXED: point data)
/// \param arrayName (DYNAMIC: string identifier of the input array)
///
/// See the corresponding standalone program for a usage example:
///   - standalone/PeriodicGhostsGeneration/main.cpp
///
/// See the related ParaView example state files for usage examples within a
/// VTK pipeline.
///
/// \sa ttk::PeriodicGhostsGeneration
/// \sa ttkAlgorithm

#pragma once

// VTK Module
#include <ttkPeriodicGhostsGenerationModule.h>

// VTK Includes
#include <ttkAlgorithm.h>
#include <vtkCellTypes.h>
#include <vtkCharArray.h>
#include <vtkCommand.h>
#include <vtkCommunicator.h>
#include <vtkDataSet.h>
#include <vtkExtractVOI.h>
#include <vtkImageAppend.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkInformationIntegerKey.h>
#include <vtkInformationVector.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>

/* Note on including VTK modules
 *
 * Each VTK module that you include a header from needs to be specified in this
 * module's vtk.module file, either in the DEPENDS or PRIVATE_DEPENDS (if the
 * header is included in the cpp file only) sections.
 *
 * In order to find the corresponding module, check its location within the VTK
 * source code. The VTK module name is composed of the path to the header. You
 * can also find the module name within the vtk.module file located in the same
 * directory as the header file.
 *
 * For example, vtkSphereSource.h is located in directory VTK/Filters/Sources/,
 * so its corresponding VTK module is called VTK::FiltersSources. In this case,
 * the vtk.module file would need to be extended to
 *
 * NAME
 *   ttkPeriodicGhostsGeneration
 * DEPENDS
 *   ttkAlgorithm
 *   VTK::FiltersSources
 */

class TTKPERIODICGHOSTSGENERATION_EXPORT ttkPeriodicGhostsGeneration
  : public ttkAlgorithm {

private:
  /**
   * TODO 5: Add all filter parameters only as private member variables and
   *         initialize them here.
   */
  std::string OutputArrayName{"AveragedScalarField"};
  vtkNew<vtkImageAppend> append;
  std::array<int, 6> outExtent_;
  std::array<double, 6> boundsWithoutGhosts_;
  std::array<double, 6> globalBounds_;
  bool isOutputExtentComputed_{false};

public:
  /**
   * TODO 6: Automatically generate getters and setters of filter
   *         parameters via vtkMacros.
   */
  vtkSetMacro(OutputArrayName, const std::string &);
  vtkGetMacro(OutputArrayName, std::string);

  /**
   * This static method and the macro below are VTK conventions on how to
   * instantiate VTK objects. You don't have to modify this.
   */
  static ttkPeriodicGhostsGeneration *New();
  vtkTypeMacro(ttkPeriodicGhostsGeneration, ttkAlgorithm);

protected:
  /**
   * TODO 7: Implement the filter constructor and destructor
   *         (see cpp file)
   */
  ttkPeriodicGhostsGeneration();
  ~ttkPeriodicGhostsGeneration() override = default;

  /**
   * TODO 8: Specify the input data type of each input port
   *         (see cpp file)
   */
  int FillInputPortInformation(int port, vtkInformation *info) override;
  int RequestUpdateExtent(vtkInformation *ttkNotUsed(request),
                          vtkInformationVector **inputVector,
                          vtkInformationVector *outputVector) override;

  /**
   * This method is called in GetTriangulation, if the triangulation is
   * periodic, to create ghosts specific to dealing with this type of
   * triangulation. This may add points to the dataset of a process and
   * therefore invalidates the triangulation object taken as parameter here.
   */
  int MPIPeriodicGhostPipelinePreconditioning(
    vtkInformationVector **inputVectors, vtkInformationVector *outputVector);

  int ComputeOutputExtent(vtkDataSet *input);

  int MergeImageAppendAndSlice(vtkImageData *image,
                               vtkImageData *slice,
                               vtkImageData *mergedImage,
                               int direction);

  /**
   * TODO 9: Specify the data object type of each output port
   *         (see cpp file)
   */
  int FillOutputPortInformation(int port, vtkInformation *info) override;

  /**
   * TODO 10: Pass VTK data to the base code and convert base code output to VTK
   *          (see cpp file)
   */
  int RequestData(vtkInformation *request,
                  vtkInformationVector **inputVector,
                  vtkInformationVector *outputVector) override;

  int RequestInformation(vtkInformation *request,
                         vtkInformationVector **inputVectors,
                         vtkInformationVector *outputVector) override;
};