// stereo3d - SPIR-V stereo patcher (see spirv_patch.h)
#include "spirv_patch.h"

#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace stereo3d {

namespace {

// --- SPIR-V constants we need -------------------------------------------------
enum : uint32_t {
  OpSourceContinued = 2, OpSource = 3, OpSourceExtension = 4, OpName = 5, OpMemberName = 6,
  OpString = 7, OpLine = 8, OpExtension = 10, OpExtInstImport = 11, OpMemoryModel = 14,
  OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
  OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
  OpTypeArray = 28, OpTypeStruct = 30, OpTypePointer = 32,
  OpConstant = 43, OpConstantComposite = 44,
  OpFunction = 54, OpFunctionEnd = 56, OpVariable = 59,
  OpLoad = 61, OpStore = 62, OpAccessChain = 65,
  OpDecorate = 71, OpMemberDecorate = 72, OpDecorationGroup = 73, OpGroupDecorate = 74,
  OpGroupMemberDecorate = 75,
  OpCompositeExtract = 81, OpCompositeInsert = 82, OpConvertFToU = 109, OpUConvert = 113, OpConvertUToPtr = 120,
  OpBitcast = 124, OpIAdd = 128, OpISub = 130,
  OpFAdd = 129, OpFSub = 131, OpIMul = 132, OpFMul = 133, OpLogicalAnd = 167, OpSelect = 169, OpIEqual = 170,
  OpFOrdNotEqual = 182, OpFOrdLessThan = 184, OpFOrdGreaterThan = 186,
  OpShiftRightLogical = 194, OpShiftLeftLogical = 196, OpBitwiseOr = 197, OpBitwiseXor = 198,
  OpBitwiseAnd = 199,
  OpEmitVertex = 218, OpEmitStreamVertex = 220,
  OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249, OpBranchConditional = 250, OpReturn = 253,
  OpModuleProcessed = 330, OpExecutionModeId = 331, OpDecorateId = 332,
  OpDecorateString = 5632, OpMemberDecorateString = 5633,
};
enum : uint32_t {
  CapInt64 = 11, CapPhysicalStorageBufferAddresses = 5347,
  AddrLogical = 0, AddrPhysicalStorageBuffer64 = 5348,
  StorageOutput = 3, StoragePhysicalStorageBuffer = 5349,
  DecBlock = 2, DecBuiltIn = 11, DecOffset = 35, BuiltInPosition = 0,
  MemAligned = 2,
};

struct Inst {
  size_t off;   // word offset
  uint32_t op;
  uint32_t wc;  // word count
};

std::string ReadString(const uint32_t* w, uint32_t maxWords) {
  std::string s;
  const char* c = reinterpret_cast<const char*>(w);
  size_t maxBytes = size_t(maxWords) * 4;
  for (size_t i = 0; i < maxBytes && c[i]; ++i) s.push_back(c[i]);
  return s;
}

uint32_t StringWords(const std::string& s) { return uint32_t(s.size() / 4 + 1); }

void Emit(std::vector<uint32_t>& o, uint32_t op, std::initializer_list<uint32_t> operands) {
  o.push_back(((uint32_t(operands.size()) + 1) << 16) | op);
  for (uint32_t v : operands) o.push_back(v);
}

void EmitString(std::vector<uint32_t>& o, uint32_t op, const std::string& s) {
  uint32_t n = StringWords(s);
  o.push_back(((n + 1) << 16) | op);
  std::vector<uint32_t> words(n, 0);
  std::memcpy(words.data(), s.data(), s.size());
  o.insert(o.end(), words.begin(), words.end());
}

bool IsPreTypeSection(uint32_t op) {
  switch (op) {
    case OpCapability: case OpExtension: case OpExtInstImport: case OpMemoryModel:
    case OpEntryPoint: case OpExecutionMode: case OpExecutionModeId:
    case OpString: case OpSourceExtension: case OpSource: case OpSourceContinued:
    case OpName: case OpMemberName: case OpModuleProcessed:
    case OpDecorate: case OpMemberDecorate: case OpDecorationGroup: case OpGroupDecorate:
    case OpGroupMemberDecorate: case OpDecorateId: case OpDecorateString:
    case OpMemberDecorateString:
      return true;
    default:
      return false;
  }
}

bool ParseInsts(const uint32_t* code, size_t n, std::vector<Inst>& insts) {
  if (n < 5 || code[0] != 0x07230203) return false;
  size_t i = 5;
  while (i < n) {
    uint32_t wc = code[i] >> 16, op = code[i] & 0xFFFF;
    if (wc == 0 || i + wc > n) return false;
    insts.push_back({i, op, wc});
    i += wc;
  }
  return true;
}

}  // namespace

const char* PatchResultName(PatchResult r) {
  switch (r) {
    case PatchResult::Patched: return "patched";
    case PatchResult::NoPosition: return "no-position";
    case PatchResult::NoEntryPoint: return "no-entry-point";
    case PatchResult::Unsupported: return "unsupported";
    default: return "error";
  }
}

bool HasPreRasterEntryPoint(const uint32_t* code, size_t n) {
  std::vector<Inst> insts;
  if (!ParseInsts(code, n, insts)) return false;
  for (const Inst& in : insts) {
    if (in.op == OpEntryPoint && in.wc >= 3) {
      uint32_t model = code[in.off + 1];
      if (model == 0 || model == 2 || model == 3) return true;
    }
    if (in.op == OpFunction) break;
  }
  return false;
}

PatchResult PatchStereo(const uint32_t* code, size_t n, const char* entryName, uint32_t execModel,
                        uint64_t paramAddress, std::vector<uint32_t>& out, std::string* log, bool calib) {
  std::vector<Inst> insts;
  if (!ParseInsts(code, n, insts)) return PatchResult::Error;
  if (execModel != 0 && execModel != 2 && execModel != 3) return PatchResult::Unsupported;

  uint32_t bound = code[3];
  auto fresh = [&]() { return bound++; };

  // --- gather module info --------------------------------------------------------
  std::unordered_set<uint32_t> caps;
  std::unordered_set<std::string> exts;
  uint32_t entryFunc = 0;
  std::vector<uint32_t> interfaceIds;
  std::unordered_set<uint32_t> positionVars;                       // decorated var ids
  std::unordered_map<uint32_t, uint32_t> positionMembers;          // struct id -> member idx
  std::unordered_map<uint32_t, const uint32_t*> types;             // type id -> inst words
  struct Var { uint32_t type; uint32_t storage; };
  std::unordered_map<uint32_t, Var> globals;
  size_t firstFunctionIdx = insts.size();
  size_t annotationEndIdx = insts.size();
  uint32_t idBool = 0, idU32 = 0, idU64 = 0, idF32 = 0;

  for (size_t ii = 0; ii < insts.size(); ++ii) {
    const Inst& in = insts[ii];
    const uint32_t* w = code + in.off;
    if (annotationEndIdx == insts.size() && !IsPreTypeSection(in.op)) annotationEndIdx = ii;
    switch (in.op) {
      case OpCapability: caps.insert(w[1]); break;
      case OpExtension: exts.insert(ReadString(w + 1, in.wc - 1)); break;
      case OpEntryPoint: {
        if (in.wc < 4) break;
        std::string name = ReadString(w + 3, in.wc - 3);
        if (w[1] == execModel && name == entryName) {
          entryFunc = w[2];
          uint32_t nameWords = StringWords(name);
          for (uint32_t k = 3 + nameWords; k < in.wc; ++k) interfaceIds.push_back(w[k]);
        }
        break;
      }
      case OpDecorate:
        if (in.wc >= 4 && w[2] == DecBuiltIn && w[3] == BuiltInPosition) positionVars.insert(w[1]);
        break;
      case OpMemberDecorate:
        if (in.wc >= 5 && w[3] == DecBuiltIn && w[4] == BuiltInPosition)
          positionMembers[w[1]] = w[2];
        break;
      case OpTypeBool: if (!idBool) idBool = w[1]; types[w[1]] = w; break;
      case OpTypeInt:
        types[w[1]] = w;
        if (w[2] == 32 && w[3] == 0 && !idU32) idU32 = w[1];
        if (w[2] == 64 && w[3] == 0 && !idU64) idU64 = w[1];
        break;
      case OpTypeFloat:
        types[w[1]] = w;
        if (w[2] == 32 && !idF32) idF32 = w[1];
        break;
      case OpTypeVector: case OpTypeStruct: case OpTypePointer: case OpTypeArray:
        types[w[1]] = w;
        break;
      case OpVariable:
        if (firstFunctionIdx == insts.size()) globals[w[2]] = {w[1], w[3]};
        break;
      case OpFunction:
        if (firstFunctionIdx == insts.size()) firstFunctionIdx = ii;
        break;
      default: break;
    }
  }
  if (!entryFunc) return PatchResult::NoEntryPoint;
  if (firstFunctionIdx == insts.size()) return PatchResult::Error;

  // --- find the Position output ---------------------------------------------------
  uint32_t posVar = 0, posMember = UINT32_MAX, vec4Type = 0;
  auto checkVar = [&](uint32_t id) -> bool {
    auto g = globals.find(id);
    if (g == globals.end() || g->second.storage != StorageOutput) return false;
    auto pt = types.find(g->second.type);
    if (pt == types.end() || (pt->second[0] & 0xFFFF) != OpTypePointer) return false;
    uint32_t pointee = pt->second[3];
    if (positionVars.count(id)) {
      posVar = id; posMember = UINT32_MAX; vec4Type = pointee; return true;
    }
    auto st = types.find(pointee);
    if (st != types.end() && (st->second[0] & 0xFFFF) == OpTypeStruct) {
      auto pm = positionMembers.find(pointee);
      if (pm != positionMembers.end()) {
        uint32_t memberCount = (st->second[0] >> 16) - 2;
        if (pm->second < memberCount) {
          posVar = id; posMember = pm->second; vec4Type = st->second[2 + pm->second];
          return true;
        }
      }
    }
    return false;
  };
  for (uint32_t id : interfaceIds)
    if (checkVar(id)) break;
  if (!posVar) {  // SPIR-V < 1.4 may omit nothing, but be lenient
    for (auto& g : globals)
      if (checkVar(g.first)) break;
  }
  if (!posVar) return PatchResult::NoPosition;

  // vec4 of float32?
  auto vt = types.find(vec4Type);
  if (vt == types.end() || (vt->second[0] & 0xFFFF) != OpTypeVector || vt->second[3] != 4)
    return PatchResult::Unsupported;
  uint32_t floatType = vt->second[2];
  auto ft = types.find(floatType);
  if (ft == types.end() || (ft->second[0] & 0xFFFF) != OpTypeFloat || ft->second[2] != 32)
    return PatchResult::Unsupported;

  // --- allocate ids ----------------------------------------------------------------
  bool addBool = !idBool, addU32 = !idU32, addU64 = !idU64;
  if (addBool) idBool = fresh();
  if (addU32) idU32 = fresh();
  if (addU64) idU64 = fresh();
  uint32_t idStruct = fresh(), idPtrStruct = fresh(), idPtrFloat = fresh();
  uint32_t idPtrOutVec4 = (posMember != UINT32_MAX) ? fresh() : 0;
  uint32_t idAddr = fresh(), idC0 = fresh(), idC1 = fresh(), idC2 = fresh(), idOne = fresh(), idEdge = fresh();
  uint32_t idMember = (posMember != UINT32_MAX) ? fresh() : 0;

  const bool isGeometry = (execModel == 3);
  // projection samples: only at OpReturn sites (splitting a geometry shader's block around
  // OpEmitVertex could break structured control flow)
  if (isGeometry) calib = false;
  uint32_t idC3 = fresh(), idHalf = fresh(), idMHalf = fresh(), idZeroF = fresh();
  uint32_t idPtrU64 = 0, idK63 = 0, idK6 = 0, idK255 = 0, idK3 = 0, idK32 = 0, idOff64 = 0, idK7 = 0, idK18 = 0,
           idK26 = 0, idKMul = 0;
  if (calib) {
    idPtrU64 = fresh(); idK63 = fresh(); idK6 = fresh(); idK255 = fresh(); idK3 = fresh(); idK32 = fresh();
    idOff64 = fresh(); idK7 = fresh(); idK18 = fresh(); idK26 = fresh(); idKMul = fresh();
  }
  int patchSites = 0;

  // --- emit -----------------------------------------------------------------------
  out.clear();
  out.reserve(n + 256);
  out.insert(out.end(), code, code + 5);

  bool capsDone = false;
  uint32_t currentFunc = 0;
  for (size_t ii = 0; ii < insts.size(); ++ii) {
    const Inst& in = insts[ii];
    const uint32_t* w = code + in.off;

    if (!capsDone && in.op != OpCapability) {
      capsDone = true;
      if (!caps.count(CapPhysicalStorageBufferAddresses))
        Emit(out, OpCapability, {CapPhysicalStorageBufferAddresses});
      if (!caps.count(CapInt64)) Emit(out, OpCapability, {CapInt64});
      if (!exts.count("SPV_KHR_physical_storage_buffer"))
        EmitString(out, OpExtension, "SPV_KHR_physical_storage_buffer");
    }

    if (ii == annotationEndIdx) {
      Emit(out, OpDecorate, {idStruct, DecBlock});
      Emit(out, OpMemberDecorate, {idStruct, 0, DecOffset, 0});
      Emit(out, OpMemberDecorate, {idStruct, 1, DecOffset, 4});
      Emit(out, OpMemberDecorate, {idStruct, 2, DecOffset, 8});
      Emit(out, OpMemberDecorate, {idStruct, 3, DecOffset, 12});
    }

    if (ii == firstFunctionIdx) {
      if (addBool) Emit(out, OpTypeBool, {idBool});
      if (addU32) Emit(out, OpTypeInt, {idU32, 32, 0});
      if (addU64) Emit(out, OpTypeInt, {idU64, 64, 0});
      Emit(out, OpTypeStruct, {idStruct, floatType, floatType, floatType, floatType});
      Emit(out, OpTypePointer, {idPtrStruct, StoragePhysicalStorageBuffer, idStruct});
      Emit(out, OpTypePointer, {idPtrFloat, StoragePhysicalStorageBuffer, floatType});
      if (idPtrOutVec4) Emit(out, OpTypePointer, {idPtrOutVec4, StorageOutput, vec4Type});
      Emit(out, OpConstant, {idU64, idAddr, uint32_t(paramAddress & 0xFFFFFFFFu),
                             uint32_t(paramAddress >> 32)});
      Emit(out, OpConstant, {idU32, idC0, 0});
      Emit(out, OpConstant, {idU32, idC1, 1});
      Emit(out, OpConstant, {idU32, idC2, 2});
      Emit(out, OpConstant, {floatType, idEdge, 0x3F7F7CEEu});  // 0.998 (x^2 limit for HUD vertices)
      if (idMember) Emit(out, OpConstant, {idU32, idMember, posMember});
      Emit(out, OpConstant, {floatType, idOne, 0x3F800000u});
      Emit(out, OpConstant, {idU32, idC3, 3});
      Emit(out, OpConstant, {floatType, idHalf, 0x3F000000u});
      Emit(out, OpConstant, {floatType, idMHalf, 0xBF000000u});  // -0.5
      Emit(out, OpConstant, {floatType, idZeroF, 0u});
      if (calib) {
        Emit(out, OpTypePointer, {idPtrU64, StoragePhysicalStorageBuffer, idU64});
        Emit(out, OpConstant, {idU32, idK63, 63});
        Emit(out, OpConstant, {idU32, idK6, 6});
        Emit(out, OpConstant, {idU32, idK255, kCalibSlots - 1});
        Emit(out, OpConstant, {idU32, idK3, 3});
        Emit(out, OpConstant, {idU32, idK32, 32});
        Emit(out, OpConstant, {idU64, idOff64, kCalibOffset, 0});
        Emit(out, OpConstant, {idU32, idK7, 7});
        Emit(out, OpConstant, {idU32, idK18, 18});
        Emit(out, OpConstant, {idU32, idK26, 26});
        Emit(out, OpConstant, {idU32, idKMul, 0x9E3779B1u});
      }
    }

    if (in.op == OpFunction) currentFunc = w[2];
    if (in.op == OpFunctionEnd) currentFunc = 0;

    bool site = isGeometry ? (in.op == OpEmitVertex || in.op == OpEmitStreamVertex)
                           : (in.op == OpReturn && currentFunc == entryFunc);
    if (site) {
      ++patchSites;
      uint32_t ptrS = fresh(), pk = fresh(), k = fresh(), pc = fresh(), cv = fresh(), ph = fresh(), hd = fresh();
      Emit(out, OpConvertUToPtr, {idPtrStruct, ptrS, idAddr});
      // params.flags < 0: the HUD drawn over the finished scene (depth / hybrid HUD separation):
      // no stereo shift, it stays on the screen plane in both eyes
      uint32_t k0 = fresh(), pf0 = fresh(), fl0 = fresh(), noShift = fresh();
      Emit(out, OpAccessChain, {idPtrFloat, pk, ptrS, idC0});
      Emit(out, OpLoad, {floatType, k0, pk, MemAligned, 4});
      Emit(out, OpAccessChain, {idPtrFloat, pf0, ptrS, idC3});
      Emit(out, OpLoad, {floatType, fl0, pf0, MemAligned, 4});
      Emit(out, OpFOrdLessThan, {idBool, noShift, fl0, idMHalf});
      Emit(out, OpSelect, {floatType, k, noShift, idZeroF, k0});
      Emit(out, OpAccessChain, {idPtrFloat, pc, ptrS, idC1});
      Emit(out, OpLoad, {floatType, cv, pc, MemAligned, 4});
      Emit(out, OpAccessChain, {idPtrFloat, ph, ptrS, idC2});
      Emit(out, OpLoad, {floatType, hd, ph, MemAligned, 4});
      uint32_t posPtr = posVar;
      if (idPtrOutVec4) {
        posPtr = fresh();
        Emit(out, OpAccessChain, {idPtrOutVec4, posPtr, posVar, idMember});
      }
      uint32_t p = fresh(), wv = fresh(), x = fresh(), d = fresh(), s = fresh(), nx = fresh(),
               persp = fresh(), fx = fresh(), np = fresh();
      uint32_t y = fresh(), xx = fresh(), yy = fresh(), inX = fresh(), inY = fresh(), inside = fresh(),
               kh = fresh(), hx = fresh(), hudx = fresh();
      Emit(out, OpLoad, {vec4Type, p, posPtr});
      Emit(out, OpCompositeExtract, {floatType, wv, p, 3});
      Emit(out, OpCompositeExtract, {floatType, x, p, 0});
      Emit(out, OpCompositeExtract, {floatType, y, p, 1});
      // perspective geometry: x += k * (w - conv)
      Emit(out, OpFSub, {floatType, d, wv, cv});
      Emit(out, OpFMul, {floatType, s, k, d});
      Emit(out, OpFAdd, {floatType, nx, x, s});
      // 2D (w == 1): HUD vertices inside the screen get x += k * hud; full-screen passes stay put
      Emit(out, OpFMul, {floatType, xx, x, x});
      Emit(out, OpFMul, {floatType, yy, y, y});
      Emit(out, OpFOrdLessThan, {idBool, inX, xx, idEdge});
      Emit(out, OpFOrdLessThan, {idBool, inY, yy, idEdge});
      Emit(out, OpLogicalAnd, {idBool, inside, inX, inY});
      Emit(out, OpFMul, {floatType, kh, k, hd});
      Emit(out, OpFAdd, {floatType, hx, x, kh});
      Emit(out, OpSelect, {floatType, hudx, inside, hx, x});
      Emit(out, OpFOrdNotEqual, {idBool, persp, wv, idOne});
      Emit(out, OpSelect, {floatType, fx, persp, nx, hudx});
      Emit(out, OpCompositeInsert, {vec4Type, np, fx, p, 0});
      Emit(out, OpStore, {posPtr, np});
      if (calib) {
        // m = (bits(x) ^ (bits(y) >> 7) ^ bits(w) ^ (bits(z) >> 6)) * 0x9E3779B1
        // if (params.flags > 0.5 && w != 1 && (m >> 26) == 0)            (1 vertex in 64)
        //   slot[(m >> 18) & 255] = uint64(bits(w)) | (uint64(bits(z)) << 32)
        uint32_t pf = fresh(), fl = fresh(), on = fresh(), wb = fresh(), lo = fresh(), loz = fresh(),
                 c1 = fresh(), cond = fresh(), lStore = fresh(), lMerge = fresh();
        Emit(out, OpAccessChain, {idPtrFloat, pf, ptrS, idC3});
        Emit(out, OpLoad, {floatType, fl, pf, MemAligned, 4});
        Emit(out, OpFOrdGreaterThan, {idBool, on, fl, idHalf});
        // pick ~1/64 of the vertices (hash of x, y, w bits: varies even on flat geometry)
        uint32_t xb = fresh(), yb = fresh(), h1 = fresh(), h2 = fresh(), hh = fresh(), zv = fresh(), zb = fresh(),
                 zs = fresh();
        Emit(out, OpBitcast, {idU32, wb, wv});
        Emit(out, OpBitcast, {idU32, xb, x});
        Emit(out, OpBitcast, {idU32, yb, y});
        Emit(out, OpCompositeExtract, {floatType, zv, p, 2});
        Emit(out, OpBitcast, {idU32, zb, zv});
        Emit(out, OpShiftRightLogical, {idU32, zs, zb, idK6});
        uint32_t ys = fresh(), h3 = fresh(), hm = fresh();
        Emit(out, OpShiftRightLogical, {idU32, ys, yb, idK7});
        Emit(out, OpBitwiseXor, {idU32, h1, xb, ys});
        Emit(out, OpBitwiseXor, {idU32, h2, h1, wb});
        Emit(out, OpBitwiseXor, {idU32, h3, h2, zs});
        // multiply (golden ratio) and use the top bits: every input bit matters
        Emit(out, OpIMul, {idU32, hm, h3, idKMul});
        Emit(out, OpShiftRightLogical, {idU32, hh, hm, idK18});
        Emit(out, OpShiftRightLogical, {idU32, lo, hm, idK26});
        Emit(out, OpIEqual, {idBool, loz, lo, idC0});
        Emit(out, OpLogicalAnd, {idBool, c1, on, persp});
        Emit(out, OpLogicalAnd, {idBool, cond, c1, loz});
        Emit(out, OpSelectionMerge, {lMerge, 0});
        Emit(out, OpBranchConditional, {cond, lStore, lMerge});
        Emit(out, OpLabel, {lStore});
        uint32_t sh = fresh(), slot = fresh(), so = fresh(), so64 = fresh(),
                 a1 = fresh(), a2 = fresh(), ptr = fresh(), w64 = fresh(), z64 = fresh(), zh = fresh(), val = fresh();
        // slot = group * 64 + hash (64 slots per group): params.flags = 1 + group, the group being
        // the number of times the scene depth was cleared so far in the frame (scenes drawn in
        // several stages with different cameras: only the last stage is in the depth copy)
        uint32_t g = fresh(), g1 = fresh(), g2 = fresh(), gs = fresh();
        Emit(out, OpBitwiseAnd, {idU32, sh, hh, idK63});
        Emit(out, OpConvertFToU, {idU32, g, fl});
        Emit(out, OpISub, {idU32, g1, g, idC1});
        Emit(out, OpBitwiseAnd, {idU32, g2, g1, idK3});
        Emit(out, OpShiftLeftLogical, {idU32, gs, g2, idK6});
        Emit(out, OpBitwiseOr, {idU32, slot, gs, sh});
        Emit(out, OpShiftLeftLogical, {idU32, so, slot, idK3});
        Emit(out, OpUConvert, {idU64, so64, so});
        Emit(out, OpIAdd, {idU64, a1, idAddr, idOff64});
        Emit(out, OpIAdd, {idU64, a2, a1, so64});
        Emit(out, OpConvertUToPtr, {idPtrU64, ptr, a2});
        Emit(out, OpUConvert, {idU64, w64, wb});
        Emit(out, OpUConvert, {idU64, z64, zb});
        Emit(out, OpShiftLeftLogical, {idU64, zh, z64, idK32});
        Emit(out, OpBitwiseOr, {idU64, val, w64, zh});
        Emit(out, OpStore, {ptr, val, MemAligned, 8});
        Emit(out, OpBranch, {lMerge});
        Emit(out, OpLabel, {lMerge});
      }
    }

    if (in.op == OpMemoryModel && w[1] == AddrLogical) {
      Emit(out, OpMemoryModel, {AddrPhysicalStorageBuffer64, w[2]});
      continue;
    }
    out.insert(out.end(), w, w + in.wc);
  }

  if (patchSites == 0) return PatchResult::NoPosition;
  out[3] = bound;
  if (log) *log = "sites=" + std::to_string(patchSites);
  return PatchResult::Patched;
}

}  // namespace stereo3d
