/// \ingroup base
/// \class ttk::LinkedList
/// \author Eve Le Guillou <eve.le-guillou@lip6.fr>
/// \date Mars 2022.

#include <array>
#include <list>

#pragma once

namespace ttk {
  template <typename datatype, int size>
  class LinkedList {
  public:
    std::list<std::array<datatype, size>> list;
    int numberOfElement;

    LinkedList()
      : list(std::list<std::array<datatype, size>>({})), numberOfElement(0) {
    }

    datatype *addArrayElement(datatype element) {
      numberOfElement = numberOfElement % size;
      if(numberOfElement == 0) {
        this->list.push_back(std::array<datatype, size>({}));
      }
      this->list.back().at(numberOfElement) = element;
      this->numberOfElement++;
      return &(this->list.back().at(numberOfElement - 1));
    }
  };
} // namespace ttk
