// stereo3d - SPIR-V stereo patcher
// Injects   pos.x += k * (pos.w - conv)   (skipped when pos.w == 1.0, i.e. HUD /
// fullscreen passes) into the last pre-rasterization stage of a pipeline.
// k and conv are read at runtime from a GPU buffer through a hard-coded
// buffer-device-address, so the eye / separation / convergence can change
// every frame without touching the game's descriptors or pipeline layouts.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stereo3d {

enum class PatchResult { Patched, NoPosition, NoEntryPoint, Unsupported, Error };

// execModel: SPIR-V ExecutionModel (0 = Vertex, 2 = TessellationEvaluation, 3 = Geometry)
// calib: also store (clip z, clip w) samples of some perspective vertices into the parameter
// buffer (slots at paramAddress + kCalibOffset, 8 bytes each) while params[3] > 0.5, so the layer
// can recover the projection's depth mapping (needs vertexPipelineStoresAndAtomics).
constexpr uint32_t kCalibOffset = 64;
constexpr uint32_t kCalibSlots = 256;
PatchResult PatchStereo(const uint32_t* code, size_t wordCount, const char* entryName,
                        uint32_t execModel, uint64_t paramAddress,
                        std::vector<uint32_t>& out, std::string* log = nullptr, bool calib = false);

// Quick scan: does this module contain an entry point that could need patching?
bool HasPreRasterEntryPoint(const uint32_t* code, size_t wordCount);

const char* PatchResultName(PatchResult r);

}  // namespace stereo3d
