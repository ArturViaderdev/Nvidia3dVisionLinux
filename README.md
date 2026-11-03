<a id="english"></a>

# 3dvisionlinux — NVIDIA 3D Vision shutter glasses for Vulkan games on Linux

**English** · [Español](#español)

A Vulkan layer that turns Linux games into **real geometric stereoscopic 3D** and shows them **frame-sequentially** on a 120 Hz monitor with **NVIDIA 3D Vision** shutter glasses and their USB IR emitter: 1920x1080 at 120 Hz, one refresh per eye, **60 fps per eye**.

- Each eye is a full render with its own camera (shader patching, like 3D Vision's automatic mode), not a depth-buffer reprojection.
- Works with every game that reaches Vulkan: Windows games on Proton/Wine (DX9/10/11 through DXVK, DX12 through VKD3D-Proton, Vulkan) and native Linux Vulkan games.
- Drives the emitter directly through the kernel's usbfs (no libusb, no NVIDIA stereo driver needed). Any GPU vendor works; what matters is a monitor running at 100/110/120 Hz.
- Built on the same engine as [stereo3d](https://github.com/ArturViaderdev/Stereo3DSBSLinux) (SBS output); this project replaces the SBS composition with a frame-sequential presenter and the 3D Vision synchronisation.

> **Status: experimental.** Everything except the real hardware has been tested (build, layer on a software Vulkan driver, synchronisation model in simulation, firmware extraction). The first runs with your glasses are the real test: follow "First run" below and check the log.

## Requirements

- NVIDIA 3D Vision (or 3D Vision 2) glasses and the **USB IR emitter** (USB id `0955:0007`). The RF "3D Vision Pro" kit is not supported.
- A monitor that runs **1920x1080 at 120 Hz** (100 and 110 Hz also work). 3D Vision / LightBoost monitors give the least ghosting.
- **X11 session**, game in **fullscreen** (or borderless fullscreen) on that monitor, desktop compositing off while playing (KDE: Alt+Shift+F12). Frame-sequential 3D needs one flip per refresh; a compositor breaks that.
- `g++` (C++17) to build. Nothing else: Vulkan headers are included and libusb is not needed.

## Installation

```sh
./install.sh               # layer + launcher + 3dvision-tool, for your user
sudo ./install-udev.sh     # once: lets your user talk to the emitter (then replug it)
```

### The emitter firmware (once)

The emitter needs a firmware that NVIDIA ships only inside its Windows driver (`nvstusb.sys`); it cannot be distributed here. Get the file from:

- a Windows installation with the 3D Vision driver: `C:\Windows\System32\drivers\nvstusb.sys`, or
- NVIDIA's standalone 3D Vision driver package for Windows (e.g. 425.31 or 452.06): extract it with `7z x <package>.exe -ofolder`.

Then:

```sh
3dvision-tool extract-fw path/to/nvstusb.sys     # or the whole extracted folder
3dvision-tool status                              # emitter found, permissions OK, firmware OK?
3dvision-tool test 10                             # glasses switch on for 10 s (no monitor sync yet)
```

The firmware is saved to `~/.config/3dvision/nvstusb.fw` and the layer loads it into the emitter automatically every time it is plugged in (the green LED turns on).

### Monitor at 120 Hz

```sh
tools/set-120hz.sh             # highest resolution available at 120 Hz (e.g. 2560x1440)
tools/set-120hz.sh DP-0 120    # or name the output / rate
```

## Launch options

Steam → game properties → General → Launch options:

```
/home/YOUR_USER/.local/bin/3dvision-run %command%
```

(`ENABLE_3DVISION=1` plus DXVK's own vsync/limiter off.) Native games: `3dvision-run ./game`.

| Case | Add in front |
|---|---|
| Games that read back from the GPU every frame (EDF5/EDF6), emulators | `NV3D_MODE=alternate` |
| Force replay without automatic fallback | `NV3D_MODE=replay NV3D_AUTO_SWITCH=0` |
| Sync / ghosting test pattern from the start | `NV3D_TEST_PATTERN=1` |
| Vulkan 1.0 apps (older native games) | `NV3D_LEGACY=1` |

In-game settings: **1920x1080 fullscreen**, **no TAA/DLSS/FSR2+/frame generation/motion blur** (they blend frames and you see double), FXAA/SMAA are fine.

## First run (tuning the glasses)

1. Start a game (or any Vulkan program fullscreen, e.g. `NV3D_PATCH=0 3dvision-run vkcube`) and press **Ctrl+Shift+F10**: test pattern, left eye white, right eye black.
2. Close your right eye: you should see **white**. Close your left eye: **black**. If it is the other way round press **Ctrl+Shift+End** (shutter invert).
3. With one eye closed, press **Ctrl+Shift+Insert / Ctrl+Shift+Delete** (shutter delay ±250 µs) until the eye that should see black sees the least grey ghost.
4. Ctrl+Shift+F10 again to go back to the game and **Ctrl+Shift+F9** to save. The delay is a property of your monitor: copy `emitter_delay_us` / `emitter_invert` into `~/.config/3dvision/default.conf` so every game uses it.
5. Then the usual depth tuning: F3/F4 separation, F5/F6 convergence (also the emitter's wheel; the emitter's button toggles 3D).

## Modes

- **replay** (default): the game renders each frame once; the layer submits the same commands again with the other eye's camera. A game at 60 fps gives **60 fps per eye** with both eyes at the same instant. Twice the GPU work. If the game needs GPU results mid-frame, those frames are 2D and the layer falls back to alternate by itself.
- **alternate**: one game frame per eye; the game must run at 120 fps for 60 per eye.

Either way the display always gets left, right, left, right… at the monitor's refresh. If the game is slower, the last stereo pair is repeated: the eye order never breaks, you only see a lower frame rate.

## In-game hotkeys (Ctrl + Shift + …)

| Key | Action |
|---|---|
| F3 / F4 | less / more separation |
| F5 / F6 | less / more convergence |
| F1 / F2 | HUD closer / farther |
| F7 | swap eyes (image) |
| F8 | 3D on/off |
| F9 | save this game's profile |
| F10 | test pattern on/off |
| F11 | capture: game image + left + right eye (PPM) |
| Insert / Delete | shutter delay +250 / −250 µs |
| End | shutter invert |
| PgUp / PgDn / Home | zoom in / out / reset |

Emitter: wheel = separation, wheel with button pressed = convergence, button = 3D on/off.

## Configuration

`~/.config/3dvision/default.conf` (see `default.conf.example`), per-game `~/.config/3dvision/<Game.exe>.conf` (F9), and `NV3D_<KEY>=value` environment variables for any key (`NV3D_SEPARATION`, `NV3D_EMITTER_DELAY_US`…). Short aliases: `NV3D_SEP`, `NV3D_CONV`, `NV3D_SWAP`, `NV3D_HUD`, `NV3D_DELAY`.

## Logs and diagnostics

`~/.config/3dvision/logs/<Game>.log`:

- `presenter queue: own queue, present wait: yes` — best case: the presenter has its own GPU queue and exact flip timing (`VK_KHR_present_wait`). Without present wait the eye sync comes from vsync timing; if the eyes look swapped use Ctrl+Shift+End.
- `display 120.0 Hz (60.0 per eye) | game pairs shown 60.0/s` — every 5 s.
- `3D Vision: emitter ON | display 119.982 Hz (locked) | ... dropped display frames N` — every 10 s. `dropped display frames` should stay near 0; if it grows, the display is missing refreshes (compositor on, window not fullscreen, another monitor driving vsync).
- `WARNING the display runs at 60 Hz` — the monitor is not at 100–120 Hz.
- `the emitter needs its firmware` / `cannot open /dev/bus/usb/...` — see Installation.

`3dvision-tool status` checks the emitter outside games.

## How it works

1. **Shader patching**: every vertex/tessellation/geometry shader shifts geometry by depth, `x += eye · separation · (w − convergence)`.
2. **Virtual swapchain**: the game renders into images owned by the layer. On each present they become one eye (alternate) or both eyes (replay) of a stereo pair (triple buffered).
3. **Presenter thread** on its own Vulkan queue owns the real swapchain (FIFO, vsync) and presents left/right on every refresh, so the game's GPU work never delays a flip.
4. **Synchronisation**: the presenter reports which eye reached the screen at each vblank (`vkWaitForPresentKHR`); a PLL models the display clock (≈20 µs jitter in simulation) and a high-priority thread sends the emitter the eye command at every vblank (+ your delay). Emitter protocol from [libnvstusb](https://github.com/eruffaldi/libnvstusb) (Bjoern Paetzel, Johann Baudy).

## Known limitations

- OpenGL-only games: no. Mesh-shader geometry stays flat.
- Wayland: not tested (X11 recommended).
- Ghosting depends on the monitor; non-LightBoost panels show more.
- Replay: effects that advance each time they are drawn may differ between eyes.

## Layout

```
src/layer.cpp          Vulkan layer: virtual swapchain, replay/alternate, stereo pairs, presenter thread
src/emitter.cpp        3D Vision emitter (usbfs), firmware, vblank PLL and sync thread
src/spirv_patch.cpp    SPIR-V patcher (stereo shift)
src/config.cpp         configuration, profiles, logs, hotkeys
src/output.cpp         optional own output window (X11)
tools/3dvision-tool    firmware extraction, status, glasses test
tools/3dvision-run     launcher;  tools/set-120hz.sh  monitor mode
udev/                  udev rule (install-udev.sh)
tests/                 Vulkan test app, sync simulation
```

License note: the emitter firmware belongs to NVIDIA and is never included; you extract it from your own copy of the driver.

---

<a id="español"></a>

# 3dvisionlinux — gafas NVIDIA 3D Vision para juegos Vulkan en Linux

[English](#english) · **Español**

Capa Vulkan que convierte los juegos de Linux en **3D estereoscópico geométrico real** y los muestra en **secuencia de fotogramas** en un monitor de 120 Hz con las gafas de obturación **NVIDIA 3D Vision** y su emisor USB de infrarrojos: 1920x1080 a 120 Hz, un refresco para cada ojo, **60 fps por ojo**.

- Cada ojo es un render completo con su propia cámara (parcheo de shaders, como el modo automático de 3D Vision), no una reproyección del buffer de profundidad.
- Funciona con todo juego que llegue a Vulkan: juegos de Windows con Proton/Wine (DX9/10/11 vía DXVK, DX12 vía VKD3D-Proton, Vulkan) y juegos nativos Vulkan.
- Controla el emisor directamente con usbfs del kernel (sin libusb ni driver estéreo de NVIDIA). Vale cualquier marca de GPU; lo importante es un monitor a 100/110/120 Hz.
- Usa el mismo motor que [stereo3d](https://github.com/ArturViaderdev/Stereo3DSBSLinux) (salida SBS); aquí la composición SBS se sustituye por un presentador secuencial y la sincronización 3D Vision.

> **Estado: experimental.** Todo salvo el hardware real está probado (compilación, la capa sobre un driver Vulkan por software, el modelo de sincronización en simulación y la extracción del firmware). La prueba de verdad son las primeras partidas con tus gafas: sigue "Primera prueba" y mira el registro.

## Requisitos

- Gafas NVIDIA 3D Vision (o 3D Vision 2) y el **emisor USB de infrarrojos** (id USB `0955:0007`). El kit por radio "3D Vision Pro" no es compatible.
- Un monitor que funcione a **1920x1080 a 120 Hz** (100 y 110 Hz también valen). Los monitores 3D Vision / LightBoost dan menos fantasma.
- **Sesión X11**, juego a **pantalla completa** (o ventana sin bordes a pantalla completa) en ese monitor y composición del escritorio desactivada mientras juegas (KDE: Alt+Shift+F12). El 3D secuencial necesita un cambio de imagen por refresco; un compositor lo rompe.
- `g++` (C++17) para compilar. Nada más: las cabeceras de Vulkan van incluidas y no hace falta libusb.

## Instalación

```sh
./install.sh               # capa + lanzador + 3dvision-tool, para tu usuario
sudo ./install-udev.sh     # una vez: permite a tu usuario usar el emisor (luego desenchúfalo y enchúfalo)
```

### El firmware del emisor (una vez)

El emisor necesita un firmware que NVIDIA solo incluye dentro de su driver de Windows (`nvstusb.sys`); no se puede distribuir aquí. Saca el archivo de:

- una instalación de Windows con el driver 3D Vision: `C:\Windows\System32\drivers\nvstusb.sys`, o
- el paquete independiente del driver 3D Vision para Windows (p. ej. 425.31 o 452.06): descomprímelo con `7z x <paquete>.exe -ocarpeta`.

Después:

```sh
3dvision-tool extract-fw ruta/a/nvstusb.sys       # o la carpeta descomprimida entera
3dvision-tool status                              # ¿emisor encontrado, permisos y firmware bien?
3dvision-tool test 10                             # las gafas se activan 10 s (aún sin sincronizar con el monitor)
```

El firmware se guarda en `~/.config/3dvision/nvstusb.fw` y la capa lo carga en el emisor automáticamente cada vez que se enchufa (se enciende el LED verde).

### Monitor a 120 Hz

```sh
tools/set-120hz.sh             # la mayor resolucion disponible a 120 Hz (p. ej. 2560x1440)
tools/set-120hz.sh DP-0 120    # o indica la salida / frecuencia
```

## Parámetros de lanzamiento

Steam → propiedades del juego → General → Opciones de lanzamiento:

```
/home/TU_USUARIO/.local/bin/3dvision-run %command%
```

(`ENABLE_3DVISION=1` y el vsync/limitador propios de DXVK desactivados). Juegos nativos: `3dvision-run ./juego`.

| Caso | Añadir delante |
|---|---|
| Juegos que leen de la GPU en cada fotograma (EDF5/EDF6), emuladores | `NV3D_MODE=alternate` |
| Forzar replay sin cambio automático | `NV3D_MODE=replay NV3D_AUTO_SWITCH=0` |
| Patrón de prueba de sincronización / fantasma desde el inicio | `NV3D_TEST_PATTERN=1` |
| Apps Vulkan 1.0 (juegos nativos antiguos) | `NV3D_LEGACY=1` |

Ajustes del juego: **1920x1080 pantalla completa**, **sin TAA/DLSS/FSR2+/generación de fotogramas/desenfoque de movimiento** (mezclan fotogramas y se ve doble); FXAA/SMAA sí.

## Primera prueba (ajustar las gafas)

1. Arranca un juego (o cualquier programa Vulkan a pantalla completa, p. ej. `NV3D_PATCH=0 3dvision-run vkcube`) y pulsa **Ctrl+Shift+F10**: patrón de prueba, ojo izquierdo blanco, derecho negro.
2. Cierra el ojo derecho: debes ver **blanco**. Cierra el izquierdo: **negro**. Si es al revés pulsa **Ctrl+Shift+Fin** (invertir obturador).
3. Con un ojo cerrado, pulsa **Ctrl+Shift+Insert / Ctrl+Shift+Supr** (retardo del obturador ±250 µs) hasta que el ojo que debe ver negro vea el mínimo fantasma gris.
4. Ctrl+Shift+F10 otra vez para volver al juego y **Ctrl+Shift+F9** para guardar. El retardo depende de tu monitor: copia `emitter_delay_us` / `emitter_invert` a `~/.config/3dvision/default.conf` para que valga en todos los juegos.
5. Luego el ajuste de profundidad de siempre: F3/F4 separación, F5/F6 convergencia (también con la rueda del emisor; su botón activa/desactiva el 3D).

## Modos

- **replay** (por defecto): el juego dibuja cada fotograma una vez y la capa vuelve a enviar los mismos comandos con la cámara del otro ojo. Un juego a 60 fps da **60 fps por ojo** y los dos ojos ven el mismo instante. Doble trabajo para la GPU. Si el juego necesita resultados de la GPU a mitad de fotograma, esos fotogramas salen en 2D y la capa pasa sola a alternate.
- **alternate**: un fotograma del juego por ojo; el juego necesita 120 fps para 60 por ojo.

En los dos casos la pantalla recibe siempre izquierdo, derecho, izquierdo, derecho… al ritmo del monitor. Si el juego va más lento se repite el último par: el orden de los ojos nunca se rompe, solo ves menos fps.

## Teclas en juego (Ctrl + Shift + …)

| Tecla | Acción |
|---|---|
| F3 / F4 | menos / más separación |
| F5 / F6 | menos / más convergencia |
| F1 / F2 | HUD más cerca / más lejos |
| F7 | invertir ojos (imagen) |
| F8 | 3D sí/no |
| F9 | guardar el perfil del juego |
| F10 | patrón de prueba sí/no |
| F11 | captura: imagen del juego + ojo izquierdo + derecho (PPM) |
| Insert / Supr | retardo del obturador +250 / −250 µs |
| Fin | invertir obturador |
| RePág / AvPág / Inicio | zoom más / menos / restablecer |

Emisor: rueda = separación, rueda con el botón pulsado = convergencia, botón = 3D sí/no.

## Configuración

`~/.config/3dvision/default.conf` (ver `default.conf.example`), perfil por juego `~/.config/3dvision/<Juego.exe>.conf` (F9) y variables `NV3D_<CLAVE>=valor` para cualquier clave (`NV3D_SEPARATION`, `NV3D_EMITTER_DELAY_US`…). Alias cortos: `NV3D_SEP`, `NV3D_CONV`, `NV3D_SWAP`, `NV3D_HUD`, `NV3D_DELAY`.

## Registros y diagnóstico

`~/.config/3dvision/logs/<Juego>.log`:

- `presenter queue: own queue, present wait: yes` — el mejor caso: el presentador tiene su propia cola de GPU y el momento exacto de cada cambio de imagen (`VK_KHR_present_wait`). Sin present wait la sincronización sale del ritmo del vsync; si los ojos se ven cambiados usa Ctrl+Shift+Fin.
- `display 120.0 Hz (60.0 per eye) | game pairs shown 60.0/s` — cada 5 s.
- `3D Vision: emitter ON | display 119.982 Hz (locked) | ... dropped display frames N` — cada 10 s. `dropped display frames` debe quedarse cerca de 0; si sube, la pantalla se salta refrescos (compositor activo, ventana no a pantalla completa, otro monitor marcando el vsync).
- `WARNING the display runs at 60 Hz` — el monitor no está a 100–120 Hz.
- `the emitter needs its firmware` / `cannot open /dev/bus/usb/...` — ver Instalación.

`3dvision-tool status` comprueba el emisor fuera de los juegos.

## Cómo funciona

1. **Parcheo de shaders**: cada shader de vértices/teselación/geometría desplaza la geometría según la profundidad, `x += ojo · separación · (w − convergencia)`.
2. **Swapchain virtual**: el juego dibuja en imágenes de la capa. En cada present pasan a ser un ojo (alternate) o los dos (replay) de un par estéreo (triple búfer).
3. **Hilo presentador** con su propia cola Vulkan: es dueño del swapchain real (FIFO, vsync) y presenta izquierdo/derecho en cada refresco, así el trabajo de GPU del juego nunca retrasa un cambio de imagen.
4. **Sincronización**: el presentador informa de qué ojo llega a la pantalla en cada vblank (`vkWaitForPresentKHR`); un PLL modela el reloj del monitor (≈20 µs de variación en simulación) y un hilo de alta prioridad envía al emisor la orden de ojo en cada vblank (+ tu retardo). Protocolo del emisor de [libnvstusb](https://github.com/eruffaldi/libnvstusb) (Bjoern Paetzel, Johann Baudy).

## Limitaciones conocidas

- Juegos solo OpenGL: no. La geometría de mesh shaders queda plana.
- Wayland: sin probar (se recomienda X11).
- El fantasma depende del monitor; los paneles sin LightBoost muestran más.
- Replay: efectos que avanzan cada vez que se dibujan pueden diferir entre ojos.

## Estructura

```
src/layer.cpp          capa Vulkan: swapchain virtual, replay/alternate, pares estéreo, hilo presentador
src/emitter.cpp        emisor 3D Vision (usbfs), firmware, PLL del vblank e hilo de sincronización
src/spirv_patch.cpp    parcheador SPIR-V (desplazamiento estéreo)
src/config.cpp         configuración, perfiles, registros, teclas
src/output.cpp         ventana propia opcional (X11)
tools/3dvision-tool    extracción de firmware, estado, prueba de gafas
tools/3dvision-run     lanzador;  tools/set-120hz.sh  modo del monitor
udev/                  regla udev (install-udev.sh)
tests/                 app de prueba Vulkan, simulación de la sincronización
```

Nota de licencia: el firmware del emisor es de NVIDIA y nunca se incluye; lo extraes de tu propia copia del driver.
