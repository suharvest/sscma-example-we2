override SCENARIO_APP_SUPPORT_LIST := $(APP_TYPE)

APPL_DEFINES += -DSSCMA
APPL_DEFINES += -DSSCMA_FACE
APPL_DEFINES += -DIP_xdma
APPL_DEFINES += -DEVT_DATAPATH

APPL_DEFINES += -DDBG_MORE -fno-threadsafe-statics -std=c++17

EVENTHANDLER_SUPPORT = event_handler
EVENTHANDLER_SUPPORT_LIST += evt_datapath

##
# library support feature
# Add new library here
# The source code should be loacted in ~\library\{lib_name}\
##
# Include sscma_micro via library.mk (includes drivers directory)
LIB_SEL = pwrmgmt tflmtag2412_u55tag2411 spi_eeprom sensordp fatfs img_proc spi_ptl hxevent sscma_micro
##
# middleware support feature
# Add new middleware here
# The source code should be loacted in ~\middleware\{mid_name}\
##
MID_SEL = fatfs
FATFS_PORT_LIST = mmc_spi
CMSIS_DRIVERS_LIST = SPI

override HX_TFM := ON
override OS_SEL := freertos_10_5_1
override OS_HAL := n
override TRUSTZONE := y
override MPU := n
override TRUSTZONE_TYPE := security
override TRUSTZONE_FW_TYPE := 1
override CIS_SEL := HM_COMMON
override EPII_USECASE_SEL := drv_user_defined

CIS_SUPPORT_INAPP = cis_sensor
CIS_SUPPORT_INAPP_MODEL = cis_ov5647

ifeq ($(strip $(TOOLCHAIN)), arm)
override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/sscma_face.sct
else#TOOLChain
ifeq ($(TARGET), SENSECAP_WATCHER)
	override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/linker/watcher.ld
else ifeq ($(TARGET), GROVE_VISION_AI_V2)
	override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/linker/grove.ld
else ifeq ($(TARGET), SENSECAP_A1102)
	override LINKER_SCRIPT_FILE := $(SCENARIO_APP_ROOT)/$(APP_TYPE)/linker/a1102.ld
endif


endif

##
# Add new external device here
# The source code should be located in ~\external\{device_name}\
##
#EXT_DEV_LIST +=


# library root declaration
LIBRARIES_ROOT = $(EPII_ROOT)/library

ifeq ($(APP_TYPE), EPII_SIMULATOR)
LIB_REQUIRED =
else
ifeq ($(SEMIHOST), y)
LIB_REQUIRED = common
else
LIB_REQUIRED = common clib
endif
endif

ifeq ($(LIB_CMSIS_NN_ENALBE), 1)
LIB_REQUIRED +=  cmsis_nn
endif

LIB_INCDIR =
LIB_CSRCDIR =
LIB_CXXSRCDIR =
LIB_ASMSRCDIR =

LIB_CSRCS =
LIB_CXXSRCS =
LIB_ASMSRCS =
LIB_ALLSRCS =

LIB_COBJS =
LIB_CXXOBJS =
LIB_ASMOBJS =
LIB_ALLOBJS =

LIB_DEFINES =
LIB_DEPS =
LIB_LIBS =

override LIB_SEL := $(strip $(LIB_SEL)) $(LIB_REQUIRED)
ifdef LIB_SEL
	LIB_SEL_SORTED = $(sort $(LIB_SEL))
	LIB_MKS = $(foreach LIB, $(LIB_SEL_SORTED), $(join $(LIB)/, $(LIB).mk))
	LIB_CV_MKS = $(foreach LIB, $(LIB_SEL_SORTED), $(join cv/$(LIB)/, $(LIB).mk))
	LIB_INFERENCE_ENGINE_MKS = $(foreach LIB, $(LIB_SEL_SORTED), $(join inference/$(LIB)/, $(LIB).mk))
	LIB_AUDIO_ALGO_MKS = $(foreach LIB, $(LIB_SEL_SORTED), $(join audio_algo/$(LIB)/, $(LIB).mk))
	LIB_INCLUDES = $(foreach LIB_MK, $(LIB_MKS), $(wildcard $(addprefix $(LIBRARIES_ROOT)/, $(LIB_MK))))
	LIB_INCLUDES += $(foreach LIB_CV_MK, $(LIB_CV_MKS), $(wildcard $(addprefix $(LIBRARIES_ROOT)/, $(LIB_CV_MK))))
	LIB_INCLUDES += $(foreach LIB_AUDIO_ALGO_MK, $(LIB_AUDIO_ALGO_MKS), $(wildcard $(addprefix $(LIBRARIES_ROOT)/, $(LIB_AUDIO_ALGO_MK))))
	LIB_INCLUDES += $(foreach LIB_INFERENCE_ENGINE_MK, $(LIB_INFERENCE_ENGINE_MKS), $(wildcard $(addprefix $(LIBRARIES_ROOT)/, $(LIB_INFERENCE_ENGINE_MK))))
	include $(LIB_INCLUDES)


	# include dependency files
	ifneq ($(MAKECMDGOALS),clean)
	-include $(LIB_DEPS)
	-include $(LIB_CV_DEPS)
	-include $(LIB_AUDIOALGO_DEPS)
	-include $(LIB_INFERENCE_ENGINE_DEPS)
	endif
endif

# Note: sscma_micro is now included via LIB_SEL and handled by library/library.mk
