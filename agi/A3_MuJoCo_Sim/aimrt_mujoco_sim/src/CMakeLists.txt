# Copyright (c) 2023, AgiBot Inc.
# All rights reserved

set_root_namespace("aimrt_mujoco_sim")

add_subdirectory(protocols)

add_subdirectory(module/mujoco_sim_module)

add_subdirectory(pkg/mujoco_sim_pkg)

if(AIMRT_MUJOCO_SIM_BUILD_EXAMPLES)
  add_subdirectory(examples)
  add_subdirectory(models)
endif()
