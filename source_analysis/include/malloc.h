#pragma once
// Analysis-only Linux compatibility. Allocation execution uses the Python model.
#include <stdlib.h>
void *memalign(size_t, size_t);
