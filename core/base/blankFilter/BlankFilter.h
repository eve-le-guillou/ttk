/// TODO 1: Provide your information
///
/// \ingroup base
/// \class ttk::BlankFilter
/// \author Eve Le Guillou
/// \date 2022
///
/// This module defines the %BlankFilter class that is a dummy filter that only
/// copies the input data in the output
///

#pragma once

// ttk common includes
#include <Debug.h>
#include <Triangulation.h>

namespace ttk {

  /**
   * The BlankFilter class provides no method.
   */
  class BlankFilter : virtual public Debug {

  public:
    BlankFilter();

    template <class dataType,
              class triangulationType = ttk::AbstractTriangulation>
    int computeAverages(dataType *outputData,
                        const dataType *inputData,
                        const triangulationType *triangulation) const {
      return 1; // return success
    }

  }; // BlankFilter class

} // namespace ttk
