#!/bin/sh
# Permite a tu usuario usar el emisor 3D Vision (USB 0955:0007). Necesita root una sola vez.
set -e
cd "$(dirname "$0")"
if [ "$(id -u)" != 0 ]; then exec sudo "$0" "$@"; fi
install -m 0644 udev/60-3dvision.rules /etc/udev/rules.d/60-3dvision.rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=0955 || true
echo "Regla instalada en /etc/udev/rules.d/60-3dvision.rules"
echo "Si el emisor ya estaba enchufado y '3dvision-tool status' sigue sin permisos, desenchufalo y vuelve a enchufarlo."
