/// \ingroup base
/// \class ttk::DataSetAttributes
/// \author Eve Le Guillou <eve.le-guillou@lip6.fr>
/// \date November 2021.
///
///\brief TTK base package defining integer values of point ghost types.
///
///\warning the integer values should match the values defined in the VTK class vtkDataSetAttributes
#pragma once

namespace ttk {

    namespace type {
        enum PointGhostTypes
        {
            DUPLICATEPOINT = 1, // the cell is present on multiple processors
        };
    }

} // namespace ttk