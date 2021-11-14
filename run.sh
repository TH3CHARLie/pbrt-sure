mkdir -p build
cd build
cmake ..
make -j20
./pbrt ../scenes/sure_test_scenes/staircase/scene.pbrt
