// gamestream protocol: copied from the gamecapture project (src/common/protocol.h), keep in sync.
// The 3dvision layer talks to gamestreamd as a game would (see layer.cpp, "Live streaming").
// gamestream - protocolo entre la capa Vulkan (dentro del juego) y el daemon.
// Socket UNIX abstracto, SOCK_SEQPACKET (cada mensaje llega entero), los
// descriptores de memoria GPU viajan con SCM_RIGHTS.
#pragma once
#include <stdint.h>

#define GS_SOCKET_NAME   "gamestream-capture-v1"   // namespace abstracto (\0 delante)
#define GS_PROTO_VERSION 1
#define GS_MAX_SLOTS     6

enum GsMsgType : uint32_t {
    GS_HELLO = 1,     // capa/ctl -> daemon
    GS_TEXTURES,      // capa -> daemon (+ nslots fds)
    GS_FRAME,         // capa -> daemon: el slot N tiene un frame listo
    GS_TOGGLE,        // capa -> daemon: han pulsado el hotkey
    GS_START,         // daemon -> capa: empieza a capturar
    GS_STOP,          // daemon -> capa: deja de capturar y libera recursos
    GS_RELEASE,       // daemon -> capa: ya he copiado el slot N, puedes reutilizarlo
    GS_CTL,           // ctl -> daemon: orden de línea de comandos
    GS_CTL_REPLY,     // daemon -> ctl
};

enum GsRole : uint32_t { GS_ROLE_GAME = 0, GS_ROLE_CTL = 1 };
enum GsCtlCmd : uint32_t { GS_CTL_TOGGLE = 1, GS_CTL_START, GS_CTL_STOP, GS_CTL_STATUS, GS_CTL_QUIT };

struct GsHello {
    uint32_t type;      // GS_HELLO
    uint32_t version;
    int32_t  pid;
    uint32_t role;
    char     exe[128];
};

struct GsTextures {
    uint32_t type;      // GS_TEXTURES
    uint32_t generation;
    uint32_t width, height;   // tamaño de salida (lo que se codifica)
    uint32_t pitch;           // bytes por fila en el buffer (alineado a 256)
    uint32_t nslots;
    uint64_t alloc_size;      // tamaño de la VkDeviceMemory exportada
    uint32_t dedicated;       // la memoria es "dedicated allocation"
    uint32_t has_uuid;
    uint8_t  device_uuid[16]; // VkPhysicalDeviceIDProperties::deviceUUID
    uint32_t pixfmt;          // 0 = BGRA8 (siempre por ahora)
    uint32_t stereo;          // 0 = 2D, 1 = 3D lado a lado comprimido (half SBS), 2 = lado a lado completo
                              // (añadido después: un mensaje sin este campo es 2D)
};

struct GsFrame {
    uint32_t type;      // GS_FRAME
    uint32_t generation;
    uint32_t slot;
    uint32_t pad;
    uint64_t ts_ns;     // CLOCK_MONOTONIC en el momento del present
};

struct GsStart {
    uint32_t type;      // GS_START
    uint32_t width, height;   // 0x0 = usa la resolución del juego
    uint32_t fps;
};

struct GsSimple {       // GS_TOGGLE, GS_STOP
    uint32_t type;
};

struct GsRelease {
    uint32_t type;      // GS_RELEASE
    uint32_t generation;
    uint32_t slot;
};

struct GsCtl {
    uint32_t type;      // GS_CTL
    uint32_t cmd;
};

struct GsCtlReply {
    uint32_t type;      // GS_CTL_REPLY
    char     text[480];
};

#define GS_MAX_MSG 512
