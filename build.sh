#!/bin/sh
# Compila la capa 3dvisionlinux y la herramienta del emisor. Solo necesita g++ (C++17).
# Las cabeceras de Vulkan van incluidas; no hace falta libusb (usa usbfs del kernel).
set -e
cd "$(dirname "$0")"
mkdir -p build
CXX=${CXX:-g++}
$CXX -std=c++17 -O2 -g0 -fPIC -shared -fvisibility=hidden -Wall -Wno-missing-field-initializers \
  -Ithird_party/include \
  src/layer.cpp src/spirv_patch.cpp src/spirv_screenfix.cpp src/config.cpp src/output.cpp src/emitter.cpp \
  -o build/lib3dvision_layer.so.new \
  -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,-z,defs -ldl -pthread
mv -f build/lib3dvision_layer.so.new build/lib3dvision_layer.so
# 32-bit layer for 32-bit games (old DX9/DX11 games under Proton). Needs g++-multilib;
# without it the included build/lib3dvision_layer32.so is used.
if echo 'int main(){}' | $CXX -m32 -x c++ - -o /dev/null 2>/dev/null; then
  $CXX -m32 -std=c++17 -O2 -g0 -fPIC -shared -fvisibility=hidden -Wall -Wno-missing-field-initializers \
    -Ithird_party/include \
    src/layer.cpp src/spirv_patch.cpp src/spirv_screenfix.cpp src/config.cpp src/output.cpp src/emitter.cpp \
    -o build/lib3dvision_layer32.so.new \
    -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,-z,defs -ldl -pthread
  mv -f build/lib3dvision_layer32.so.new build/lib3dvision_layer32.so
else
  echo "(no 32-bit compiler: using the included build/lib3dvision_layer32.so; sudo apt install g++-multilib to rebuild it)"
fi
$CXX -std=c++17 -O2 -Wall -Ithird_party/include tools/3dvision-tool.cpp src/emitter.cpp src/config.cpp src/spirv_patch.cpp src/spirv_screenfix.cpp \
  -o build/3dvision-tool.new -ldl -pthread
mv -f build/3dvision-tool.new build/3dvision-tool
echo "OK -> build/lib3dvision_layer.so build/lib3dvision_layer32.so build/3dvision-tool"
