#ifndef SURE_BASED_UTILITY_H
#define SURE_BASED_UTILITY_H
#include "pbrt.h"
namespace pbrt {

struct SUREBasedAuxiliaryData {
  Normal3f normal;
  Spectrum texture_value;
  // for depth we are using a relative depth computed using cur_depth / max_depth
  Float depth;
};

}

#endif