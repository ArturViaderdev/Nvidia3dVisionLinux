#!/bin/sh
# Desinstala 3dvisionlinux (deja tus perfiles en ~/.config/3dvision y la regla udev).
D="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -f "$D/vulkan/implicit_layer.d/3dvision.json" "$HOME/.local/lib/3dvision/lib3dvision_layer.so"
rmdir "$HOME/.local/lib/3dvision" 2>/dev/null
FP="$HOME/.var/app/com.valvesoftware.Steam"
rm -f "$FP/data/vulkan/implicit_layer.d/3dvision.json" "$FP/data/3dvision-lib/lib3dvision_layer.so"
rm -f "$HOME/.local/bin/3dvision-run" "$HOME/.local/bin/3dvision-tool"
echo "Desinstalado. Perfiles en ~/.config/3dvision. Regla udev: sudo rm /etc/udev/rules.d/60-3dvision.rules"
