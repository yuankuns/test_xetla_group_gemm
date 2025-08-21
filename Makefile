IPEX_HOME = /home/syk/frameworks.ai.pytorch.ipex-gpu/
CXX = icpx
CXXFLAGS = -std=c++20 -O1 -fPIC -fsycl -fsycl-targets=spir64_gen
# CXXFLAGS = -std=c++20 -O0 -g -fPIC -fsycl -fsycl-targets=spir64_gen
AOTFLAGS = -Xsycl-target-backend=spir64_gen "-device bmg-g21-a0 -options '-doubleGRF -vc-codegen -Xfinalizer -printregusage'"

ESIMD_PATH = $(CMPLR_ROOT)/include/sycl

CXXOPTS_PATH = $(CURDIR)/cxxopts
INCLUDES = -I. -I$(CXXOPTS_PATH)/include -I$(ESIMD_PATH)
LIBS = -lz -L$(LIB_DIR) -Wl,-rpath,$(LIB_DIR)

IPEX_XETLA_DIR = $(IPEX_HOME)/csrc/gpu/aten/operators/xetla/kernels
XETLA_INCLUDES = -I$(IPEX_XETLA_DIR)/include -I$(IPEX_XETLA_DIR)

SRC = groupgemm.cpp cnpy.cpp
OUT = groupgemm

all:
	$(CXX) $(CXXFLAGS) $(AOTFLAGS) $(BINDFLAGS) $(SRC) $(INCLUDES) $(XETLA_INCLUDES) $(LIBS) -o $(OUT)

clean:
	rm -f $(OUT)

