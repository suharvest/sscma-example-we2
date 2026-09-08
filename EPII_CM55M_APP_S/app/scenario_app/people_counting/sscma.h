/* Shim.
 *
 * app/main.c selects its entry point with `#ifdef SSCMA` and includes
 * "sscma.h" by that fixed name. This app keeps -DSSCMA (sscma_micro needs it)
 * but names its own header after the app, so forward the include here rather
 * than touching the shared app/main.c.
 */
#pragma once

#include "people_counting.h"
