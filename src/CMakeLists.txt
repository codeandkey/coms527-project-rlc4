# If FindPackage(Torch) is not available, make sure Torch is installed on the host system.
# Adding the build flag `-DCMAKE_PREFIX_PATH=/path/to/libtorch/install` should resolve the problem.
 
add_executable(coms527-app coms527-app.cpp)
target_link_libraries(coms527-app "${TORCH_LIBRARIES}")
set_property(TARGET coms527-app PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET coms527-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:coms527-app>)
endif (MSVC)

# Add test binary

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(coms527-test coms527-test.cpp)
target_link_libraries(coms527-test gtest_main)

include(GoogleTest)
gtest_discover_tests(coms527-test)