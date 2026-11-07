#!/bin/sh
# Instala la capa Vulkan 3dvisionlinux para tu usuario (no necesita root).
set -e
cd "$(dirname "$0")"

if command -v g++ >/dev/null 2>&1; then
  ./build.sh || echo "Aviso: no se pudo recompilar, uso los binarios incluidos."
fi
[ -f build/lib3dvision_layer.so ] || { echo "Falta build/lib3dvision_layer.so (instala g++ y ejecuta ./build.sh)"; exit 1; }

install_to() {   # $1 = data dir (contiene vulkan/implicit_layer.d), $2 = lib dir
  mkdir -p "$1/vulkan/implicit_layer.d" "$2"
  cp build/lib3dvision_layer.so "$2/"
  sed "s#@LIB@#$2/lib3dvision_layer.so#" 3dvision_layer.json.in > "$1/vulkan/implicit_layer.d/3dvision.json"
  echo "  capa -> $1/vulkan/implicit_layer.d/3dvision.json"
  if [ -f build/lib3dvision_layer32.so ]; then  # 32-bit games
    cp build/lib3dvision_layer32.so "$2/"
    sed "s#@LIB@#$2/lib3dvision_layer32.so#; s#VK_LAYER_3DVISION_linux#VK_LAYER_3DVISION_linux_32#" 3dvision_layer.json.in \
      > "$1/vulkan/implicit_layer.d/3dvision32.json"
    echo "  capa 32 bits -> $1/vulkan/implicit_layer.d/3dvision32.json"
  fi
}

echo "Instalando 3dvisionlinux..."
install_to "${XDG_DATA_HOME:-$HOME/.local/share}" "$HOME/.local/lib/3dvision"

FP="$HOME/.var/app/com.valvesoftware.Steam"
if [ -d "$FP" ]; then
  install_to "$FP/data" "$FP/data/3dvision-lib"
  echo "  (detectado Steam Flatpak)"
fi

mkdir -p "$HOME/.local/bin"
cp tools/3dvision-run "$HOME/.local/bin/3dvision-run"
cp tools/3dvision-recorder "$HOME/.local/bin/3dvision-recorder"
cp tools/3dvision-shader-scan "$HOME/.local/bin/3dvision-shader-scan"
chmod +x "$HOME/.local/bin/3dvision-run" "$HOME/.local/bin/3dvision-recorder" "$HOME/.local/bin/3dvision-shader-scan"
[ -f build/3dvision-tool ] && cp build/3dvision-tool "$HOME/.local/bin/3dvision-tool"
echo "  lanzador -> $HOME/.local/bin/3dvision-run"
echo "  herramienta del emisor -> $HOME/.local/bin/3dvision-tool"

CFG="${XDG_CONFIG_HOME:-$HOME/.config}/3dvision"
mkdir -p "$CFG"
[ -f "$CFG/default.conf" ] || cp default.conf.example "$CFG/default.conf"
if [ -d "$FP" ]; then
  mkdir -p "$FP/config/3dvision"
  [ -f "$FP/config/3dvision/default.conf" ] || cp default.conf.example "$FP/config/3dvision/default.conf"
  [ -f "$CFG/nvstusb.fw" ] && cp "$CFG/nvstusb.fw" "$FP/config/3dvision/" || true
fi

echo
echo "Comprobando el emisor:"
"$HOME/.local/bin/3dvision-tool" status || true
[ -f /etc/udev/rules.d/60-3dvision.rules ] || echo "  -> Falta la regla udev: sudo ./install-udev.sh"
[ -f "$CFG/nvstusb.fw" ] || echo "  -> Falta el firmware: 3dvision-tool extract-fw <nvstusb.sys o carpeta del driver> (ver README)"

cat <<EOT

Listo. Opciones de lanzamiento en Steam (Propiedades del juego > General):

  $HOME/.local/bin/3dvision-run %command%

Monitor a 1920x1080 120 Hz:   tools/set-120hz.sh
Prueba de sincronizacion:     NV3D_TEST_PATTERN=1 + Ctrl+Shift+F10 en juego (ver README)
Teclas: Ctrl+Shift + F3/F4 separacion | F5/F6 convergencia | F7 invertir ojos | Ins/Supr retardo obturador | F9 guardar
EOT
