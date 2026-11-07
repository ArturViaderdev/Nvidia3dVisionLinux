// stereo3d - stereo correction of screen-space position rebuilds (see spirv_patch.h)
//
// Deferred engines (Unreal Engine 4/5...) compute shadows, lights, ambient occlusion, fog... in
// full-screen passes that rebuild each pixel's position from the depth buffer:
//
//     ScreenPos = SvPosition.xy -> [-1, 1]          SceneW = linear depth read from a texture
//     Clip      = float4(ScreenPos * SceneW, SceneW, 1)
//     Shadow / World position = mul(Clip, Matrix)
//
// The geometry of each eye was drawn shifted by  x += k * (w - conv),  so the pixel's clip x is
// k * (SceneW - conv) off and the rebuilt position (shadow lookup, light vector...) lands in the
// wrong place. This pass finds  ScreenPos.x * SceneW  and replaces it by
//
//     ScreenPos.x * SceneW - k * f * (SceneW - conv)        (f = params[4]: 1 on, 0 off)
//
// How it is found: a forward data-flow pass over every function (straight-line order, per vector
// component) tags values coming from the pixel position x / y (FragCoord, invocation id), from
// texture reads and from uniform data. A product  (x from the screen, nothing from textures) *
// (texture-derived, nothing from the screen)  is a candidate; it is patched only if, further on,
// it is multiplied by a uniform-only value AND summed with the same kind of term for y: that is
// a matrix applied to (x*W, y*W, ...), the signature of the position rebuild.
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "spirv_patch.h"

namespace stereo3d {

namespace {

enum : uint32_t {
  OpExtension = 10, OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14, OpEntryPoint = 15,
  OpCapability = 17, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
  OpTypeMatrix = 24, OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43, OpFunction = 54,
  OpFunctionParameter = 55, OpFunctionEnd = 56, OpVariable = 59, OpLoad = 61, OpStore = 62,
  OpAccessChain = 65, OpInBoundsAccessChain = 66, OpPtrAccessChain = 67, OpDecorate = 71,
  OpMemberDecorate = 72, OpVectorShuffle = 79, OpCompositeConstruct = 80, OpCompositeExtract = 81,
  OpCompositeInsert = 82, OpConvertUToPtr = 120, OpFSub = 131, OpFMul = 133, OpVectorTimesScalar = 142,
  OpVectorTimesMatrix = 144, OpMatrixTimesVector = 145, OpDot = 148, OpLabel = 248,
};
enum : uint32_t {
  CapInt64 = 11, CapPhysicalStorageBufferAddresses = 5347, AddrLogical = 0, AddrPhysicalStorageBuffer64 = 5348,
  StorageUniformConstant = 0, StorageInput = 1, StorageUniform = 2, StoragePrivate = 6, StorageFunction = 7,
  StoragePushConstant = 9, StorageStorageBuffer = 12, StoragePhysicalStorageBuffer = 5349,
  DecBlock = 2, DecBuiltIn = 11, DecOffset = 35, MemAligned = 2,
  BuiltInFragCoord = 15, BuiltInGlobalInvocationId = 28, GlslFma = 50,
};

// taint bits
enum : uint8_t { SX = 1, SY = 2, TX = 4, UN = 8, PX = 16, PY = 32, XM = 64, YM = 128 };

struct Val {
  uint8_t n = 1;
  uint8_t t[4] = {0, 0, 0, 0};
  uint8_t All() const { return uint8_t(t[0] | t[1] | t[2] | t[3]); }
};

struct TypeInfo {
  uint32_t op = 0;
  uint32_t comps = 1;     // vector size (1 for scalars and everything else)
  uint32_t elem = 0;      // vector component type
  uint32_t storage = 0;   // pointer storage class
  uint32_t pointee = 0;
  uint32_t width = 0;     // float/int width
};

struct Cand {
  uint32_t id;       // product result
  uint32_t comp;     // component of the result that holds x * W
  bool vec;          // result is a vector
  uint32_t w;        // the W operand
  int wComp;         // component of W (-1: scalar)
  bool confirmed = false;
};

bool IsScreenX(uint8_t x) { return (x & SX) && !(x & (SY | TX | PX | PY)); }
bool IsScreenY(uint8_t x) { return (x & SY) && !(x & (SX | TX | PX | PY)); }
bool IsDepth(uint8_t x) { return (x & TX) && !(x & (SX | SY | PX | PY)); }
bool PureU(uint8_t x) { return (x & UN) && !(x & (SX | SY | TX | PX | PY)); }

void Emit(std::vector<uint32_t>& o, uint32_t op, std::initializer_list<uint32_t> operands) {
  o.push_back(((uint32_t(operands.size()) + 1) << 16) | op);
  for (uint32_t v : operands) o.push_back(v);
}

}  // namespace

PatchResult PatchScreenSpace(const uint32_t* code, size_t n, uint64_t paramAddress, std::vector<uint32_t>& out,
                             std::string* log) {
  if (n < 5 || code[0] != 0x07230203) return PatchResult::Error;
  struct Inst { size_t off; uint32_t op, wc; };
  std::vector<Inst> insts;
  for (size_t i = 5; i < n;) {
    uint32_t wc = code[i] >> 16, op = code[i] & 0xFFFF;
    if (wc == 0 || i + wc > n) return PatchResult::Error;
    insts.push_back({i, op, wc});
    i += wc;
  }

  std::unordered_map<uint32_t, TypeInfo> types;
  std::unordered_map<uint32_t, uint32_t> constU32, builtin, varStorage, idType;
  std::unordered_set<uint32_t> caps, uniqueVecVars;
  bool hasPsbExt = false;
  uint32_t glsl = 0, idF32 = 0, idU32 = 0, idU64 = 0;
  size_t firstFunc = insts.size(), annotEnd = insts.size();
  bool isShaderStage = false;

  for (size_t ii = 0; ii < insts.size(); ++ii) {
    const uint32_t* w = code + insts[ii].off;
    uint32_t op = insts[ii].op, wc = insts[ii].wc;
    if (annotEnd == insts.size() && op != OpCapability && op != OpExtension && op != OpExtInstImport &&
        op != OpMemoryModel && op != OpEntryPoint && op != 16 && op != 331 && !(op >= 2 && op <= 8) &&
        op != 330 && op != OpDecorate && op != OpMemberDecorate && !(op >= 73 && op <= 75) && op != 332 &&
        op != 5632 && op != 5633)
      annotEnd = ii;
    switch (op) {
      case OpCapability: caps.insert(w[1]); break;
      case OpExtension:
        if (!strncmp((const char*)(w + 1), "SPV_KHR_physical_storage_buffer", (wc - 1) * 4)) hasPsbExt = true;
        break;
      case OpExtInstImport:
        if (!strncmp((const char*)(w + 2), "GLSL.std.450", (wc - 2) * 4)) glsl = w[1];
        break;
      case OpEntryPoint:
        if (w[1] == 4 || w[1] == 5) isShaderStage = true;  // Fragment / GLCompute
        break;
      case OpDecorate:
        if (wc >= 4 && w[2] == DecBuiltIn) builtin[w[1]] = w[3];
        break;
      case OpTypeInt:
        types[w[1]] = {op, 1, 0, 0, 0, w[2]};
        if (w[2] == 32 && !w[3] && !idU32) idU32 = w[1];
        if (w[2] == 64 && !w[3] && !idU64) idU64 = w[1];
        break;
      case OpTypeFloat:
        types[w[1]] = {op, 1, 0, 0, 0, w[2]};
        if (w[2] == 32 && !idF32) idF32 = w[1];
        break;
      case OpTypeBool: types[w[1]] = {op, 1}; break;
      case OpTypeVector: types[w[1]] = {op, w[3], w[2]}; break;
      case OpTypePointer: types[w[1]] = {op, 1, 0, w[2], w[3]}; break;
      case OpConstant:
        if (types.count(w[1]) && types[w[1]].op == OpTypeInt && wc >= 4) constU32[w[2]] = w[3];
        break;
      case OpVariable:
        varStorage[w[2]] = w[3];
        idType[w[2]] = w[1];
        break;
      case OpFunction:
        if (firstFunc == insts.size()) firstFunc = ii;
        break;
      default:
        if (op >= 19 && op <= 39 && wc >= 2) types.emplace(w[1], TypeInfo{op});  // other types
        break;
    }
  }
  if (!isShaderStage || !idF32 || firstFunc == insts.size()) return PatchResult::Unsupported;

  auto comps = [&](uint32_t type) -> uint32_t {
    auto it = types.find(type);
    if (it == types.end()) return 1;
    return it->second.op == OpTypeVector ? std::min<uint32_t>(it->second.comps, 4) : 1;
  };

  // ---- data-flow -----------------------------------------------------------------------------
  std::unordered_map<uint32_t, Val> val;
  std::unordered_map<uint32_t, std::vector<uint32_t>> prov;  // id -> candidate indices behind it
  struct PtrInfo { uint32_t var; int comp; };
  std::unordered_map<uint32_t, PtrInfo> ptrs;
  std::unordered_map<uint32_t, Val> mem;                     // Function / Private variables
  std::unordered_map<uint32_t, std::vector<uint32_t>> memProv;
  std::vector<Cand> cands;

  auto get = [&](uint32_t id, uint32_t cnt) -> Val {
    Val r;
    r.n = uint8_t(cnt);
    auto it = val.find(id);
    if (it == val.end()) return r;
    const Val& v = it->second;
    if (v.n == cnt) return v;
    uint8_t u = v.n == 1 ? v.t[0] : v.All();
    for (uint32_t c = 0; c < cnt; ++c) r.t[c] = u;
    return r;
  };
  auto addProv = [&](std::vector<uint32_t>& dst, uint32_t from) {
    auto it = prov.find(from);
    if (it == prov.end()) return;
    for (uint32_t c : it->second)
      if (dst.size() < 32 && std::find(dst.begin(), dst.end(), c) == dst.end()) dst.push_back(c);
  };
  // product of two components; records candidates
  auto product = [&](uint8_t a, uint8_t b, uint32_t ida, uint32_t idb, bool bIsVec, int comp, uint32_t rid,
                     bool rvec, std::vector<uint32_t>& rprov) -> uint8_t {
    uint8_t r = uint8_t(a | b);
    for (int side = 0; side < 2; ++side) {
      uint8_t x = side ? b : a, y = side ? a : b;
      uint32_t idy = side ? ida : idb;
      bool yVec = side ? !bIsVec : bIsVec;  // (only used for the W operand)
      (void)yVec;
      if (IsScreenX(x) && IsDepth(y)) {
        r |= PX;
        Cand c{rid, uint32_t(comp < 0 ? 0 : comp), rvec, idy, -1};
        auto ty = idType.find(idy);
        if (ty != idType.end() && comps(ty->second) > 1) c.wComp = comp < 0 ? 0 : comp;
        cands.push_back(c);
        rprov.push_back(uint32_t(cands.size() - 1));
      }
      if (IsScreenY(x) && IsDepth(y)) r |= PY;
      if ((x & PX) && PureU(y)) r |= XM;
      if ((x & PY) && PureU(y)) r |= YM;
    }
    return r;
  };

  for (size_t ii = firstFunc; ii < insts.size(); ++ii) {
    const uint32_t* w = code + insts[ii].off;
    uint32_t op = insts[ii].op, wc = insts[ii].wc;
    if (op == OpFunction || op == OpFunctionEnd || op == OpLabel) continue;

    if (op == OpVariable) {  // function-local variable
      uint32_t pointee = types.count(w[1]) ? types[w[1]].pointee : 0;
      Val v;
      v.n = uint8_t(comps(pointee));
      mem[w[2]] = v;
      varStorage[w[2]] = w[3];
      idType[w[2]] = w[1];
      continue;
    }
    if (op == OpAccessChain || op == OpInBoundsAccessChain || op == OpPtrAccessChain) {
      if (wc < 4) continue;
      uint32_t base = w[3];
      PtrInfo pi{base, -1};
      auto bp = ptrs.find(base);
      if (bp != ptrs.end()) pi = bp->second;
      uint32_t first = op == OpPtrAccessChain ? 5 : 4;
      if (wc > first && bp == ptrs.end()) {
        auto c = constU32.find(w[wc - 1]);
        if (c != constU32.end() && wc == first + 1) pi.comp = int(c->second);
      }
      ptrs[w[2]] = pi;
      idType[w[2]] = w[1];
      continue;
    }
    if (op == OpStore) {
      if (wc < 3) continue;
      PtrInfo pi{w[1], -1};
      auto p = ptrs.find(w[1]);
      if (p != ptrs.end()) pi = p->second;
      auto m = mem.find(pi.var);
      if (m == mem.end()) continue;
      Val v = get(w[2], pi.comp >= 0 ? 1 : m->second.n);
      if (pi.comp >= 0 && pi.comp < m->second.n) m->second.t[pi.comp] = v.t[0];
      else if (pi.comp < 0) m->second = v;
      addProv(memProv[pi.var], w[2]);
      continue;
    }
    // everything below has (result type, result id)
    if (wc < 3 || !types.count(w[1]) || types.count(w[2])) continue;
    uint32_t rt = w[1], rid = w[2];
    idType[rid] = rt;
    uint32_t cnt = comps(rt);
    Val r;
    r.n = uint8_t(cnt);
    std::vector<uint32_t> rprov;

    switch (op) {
      case OpLoad: {
        PtrInfo pi{w[3], -1};
        auto p = ptrs.find(w[3]);
        if (p != ptrs.end()) pi = p->second;
        uint32_t sc = varStorage.count(pi.var) ? varStorage[pi.var] : 0;
        if (!varStorage.count(pi.var)) {  // pointer from OpConvertUToPtr etc.
          auto t = idType.find(w[3]);
          if (t != idType.end() && types.count(t->second)) sc = types[t->second].storage;
        }
        auto b = builtin.find(pi.var);
        if (sc == StorageInput && b != builtin.end() &&
            (b->second == BuiltInFragCoord || b->second == BuiltInGlobalInvocationId)) {
          for (uint32_t c = 0; c < cnt; ++c) {
            int src = pi.comp >= 0 ? pi.comp : int(c);
            r.t[c] = src == 0 ? SX : src == 1 ? SY : 0;
          }
        } else if (sc == StorageUniform || sc == StoragePushConstant || sc == StorageStorageBuffer ||
                   sc == StoragePhysicalStorageBuffer) {
          for (uint32_t c = 0; c < cnt; ++c) r.t[c] = UN;
        } else if (mem.count(pi.var)) {
          const Val& m = mem[pi.var];
          if (pi.comp >= 0 && pi.comp < m.n) {
            for (uint32_t c = 0; c < cnt; ++c) r.t[c] = m.t[pi.comp];
          } else if (m.n == cnt) {
            r = m;
          } else {
            uint8_t u = m.All();
            for (uint32_t c = 0; c < cnt; ++c) r.t[c] = u;
          }
          auto mp = memProv.find(pi.var);
          if (mp != memProv.end()) rprov = mp->second;
        }
        break;
      }
      case OpCompositeExtract: {
        Val s = val.count(w[3]) ? val[w[3]] : Val();
        uint32_t idx = wc > 4 ? w[4] : 0;
        uint8_t t = s.n > 1 && idx < s.n ? s.t[idx] : s.All();
        for (uint32_t c = 0; c < cnt; ++c) r.t[c] = t;
        addProv(rprov, w[3]);
        break;
      }
      case OpCompositeConstruct: {
        std::vector<uint8_t> all;
        for (uint32_t k = 3; k < wc; ++k) {
          Val s = val.count(w[k]) ? val[w[k]] : Val();
          for (uint32_t c = 0; c < s.n; ++c) all.push_back(s.t[c]);
          addProv(rprov, w[k]);
        }
        for (uint32_t c = 0; c < cnt; ++c) r.t[c] = c < all.size() ? all[c] : 0;
        break;
      }
      case OpCompositeInsert: {
        if (wc < 6) break;
        r = get(w[4], cnt);
        Val o = get(w[3], 1);
        if (w[5] < cnt) r.t[w[5]] = o.t[0];
        addProv(rprov, w[3]);
        addProv(rprov, w[4]);
        break;
      }
      case OpVectorShuffle: {
        Val a = val.count(w[3]) ? val[w[3]] : Val(), b = val.count(w[4]) ? val[w[4]] : Val();
        uint32_t na = comps(idType.count(w[3]) ? idType[w[3]] : 0);
        for (uint32_t c = 0; c < cnt && 5 + c < wc; ++c) {
          uint32_t s = w[5 + c];
          r.t[c] = s < na ? (s < a.n ? a.t[s] : a.All()) : (s - na < b.n ? b.t[s - na] : b.All());
        }
        addProv(rprov, w[3]);
        addProv(rprov, w[4]);
        break;
      }
      case OpFMul: {
        Val a = get(w[3], cnt), b = get(w[4], cnt);
        for (uint32_t c = 0; c < cnt; ++c)
          r.t[c] = product(a.t[c], b.t[c], w[3], w[4], false, cnt > 1 ? int(c) : -1, rid, cnt > 1, rprov);
        addProv(rprov, w[3]);
        addProv(rprov, w[4]);
        break;
      }
      case OpVectorTimesScalar: {
        Val a = get(w[3], cnt), b = get(w[4], 1);
        for (uint32_t c = 0; c < cnt; ++c) {
          size_t before = cands.size();
          r.t[c] = product(a.t[c], b.t[0], w[3], w[4], false, int(c), rid, true, rprov);
          for (size_t k = before; k < cands.size(); ++k)
            if (cands[k].w == w[4]) cands[k].wComp = -1;  // the scalar is W
        }
        addProv(rprov, w[3]);
        addProv(rprov, w[4]);
        break;
      }
      case OpDot: case OpMatrixTimesVector: case OpVectorTimesMatrix: {
        uint32_t vecId = op == OpMatrixTimesVector ? w[4] : w[3], other = op == OpMatrixTimesVector ? w[3] : w[4];
        Val v = val.count(vecId) ? val[vecId] : Val(), m = val.count(other) ? val[other] : Val();
        if (op == OpDot && PureU(v.All()) && !PureU(m.All())) std::swap(v, m);
        uint8_t mu = m.All(), acc = mu;
        for (uint32_t c = 0; c < v.n; ++c) {
          uint8_t x = v.t[c];
          acc |= x;
          if (PureU(mu) && (x & PX)) acc |= XM;
          if (PureU(mu) && (x & PY)) acc |= YM;
        }
        for (uint32_t c = 0; c < cnt; ++c) r.t[c] = acc;
        addProv(rprov, w[3]);
        addProv(rprov, w[4]);
        break;
      }
      default: {
        if (op == OpExtInst && wc >= 5 && w[3] == glsl && w[4] == GlslFma && wc >= 8) {
          Val a = get(w[5], cnt), b = get(w[6], cnt), c3 = get(w[7], cnt);
          for (uint32_t c = 0; c < cnt; ++c)
            r.t[c] = uint8_t(product(a.t[c], b.t[c], w[5], w[6], false, cnt > 1 ? int(c) : -1, rid, cnt > 1, rprov) |
                             c3.t[c]);
          for (uint32_t k = 5; k < 8; ++k) addProv(rprov, w[k]);
          break;
        }
        bool image = (op >= 87 && op <= 98) || (op >= 305 && op <= 315);
        if (image) {
          for (uint32_t c = 0; c < cnt; ++c) r.t[c] = TX;
          break;
        }
        // element-wise (or unknown): per component when the sizes match, else the union
        uint32_t firstOperand = op == OpExtInst ? 5 : 3;
        for (uint32_t k = firstOperand; k < wc; ++k) {
          auto it = val.find(w[k]);
          if (it == val.end()) continue;
          const Val& s = it->second;
          for (uint32_t c = 0; c < cnt; ++c) r.t[c] |= (s.n == cnt) ? s.t[c] : s.All();
          addProv(rprov, w[k]);
        }
        break;
      }
    }
    // a matrix applied to (x*W, y*W, ...): both terms meet here
    for (uint32_t c = 0; c < cnt; ++c)
      if ((r.t[c] & XM) && (r.t[c] & YM)) {
        for (uint32_t k : rprov) cands[k].confirmed = true;
        break;
      }
    val[rid] = r;
    if (!rprov.empty()) prov[rid] = std::move(rprov);
  }

  // ---- patch ---------------------------------------------------------------------------------
  std::unordered_map<uint32_t, std::vector<const Cand*>> sites;
  for (const Cand& c : cands) {
    if (!c.confirmed) continue;
    auto rt = idType.find(c.id);
    if (rt == idType.end()) continue;
    const TypeInfo& t = types[rt->second];
    uint32_t ft = t.op == OpTypeVector ? t.elem : rt->second;
    if (ft != idF32) continue;
    auto wt = idType.find(c.w);
    if (wt == idType.end()) continue;
    const TypeInfo& wti = types[wt->second];
    if ((wti.op == OpTypeVector ? wti.elem : wt->second) != idF32) continue;
    auto& v = sites[c.id];
    bool dup = false;
    for (const Cand* o : v) dup |= o->comp == c.comp;
    if (!dup) v.push_back(&c);
  }
  if (sites.empty()) return PatchResult::NoPosition;

  uint32_t bound = code[3];
  auto fresh = [&]() { return bound++; };
  bool addU32 = !idU32, addU64 = !idU64;
  if (addU32) idU32 = fresh();
  if (addU64) idU64 = fresh();
  uint32_t idStruct = fresh(), idPtrStruct = fresh(), idPtrFloat = fresh(), idAddr = fresh(), idC0 = fresh(),
           idC1 = fresh(), idC4 = fresh();

  std::unordered_map<uint32_t, uint32_t> renamed;  // product id -> new id of the uncorrected product
  for (auto& st : sites) renamed[st.first] = fresh();

  out.clear();
  out.reserve(n + 64 + sites.size() * 40);
  out.insert(out.end(), code, code + 5);
  bool capsDone = false;
  int patched = 0;
  for (size_t ii = 0; ii < insts.size(); ++ii) {
    const uint32_t* w = code + insts[ii].off;
    uint32_t op = insts[ii].op, wc = insts[ii].wc;
    if (!capsDone && op != OpCapability) {
      capsDone = true;
      if (!caps.count(CapPhysicalStorageBufferAddresses)) Emit(out, OpCapability, {CapPhysicalStorageBufferAddresses});
      if (!caps.count(CapInt64)) Emit(out, OpCapability, {CapInt64});
      if (!hasPsbExt) {
        const char s[] = "SPV_KHR_physical_storage_buffer";  // 31 chars + NUL = 8 words
        uint32_t words[8] = {};
        memcpy(words, s, sizeof(s));
        out.push_back((9u << 16) | OpExtension);
        out.insert(out.end(), words, words + 8);
      }
    }
    if (ii == annotEnd) {
      Emit(out, OpDecorate, {idStruct, DecBlock});
      for (uint32_t m = 0; m < 5; ++m) Emit(out, OpMemberDecorate, {idStruct, m, DecOffset, m * 4});
    }
    if (ii == firstFunc) {
      if (addU32) Emit(out, OpTypeInt, {idU32, 32, 0});
      if (addU64) Emit(out, OpTypeInt, {idU64, 64, 0});
      Emit(out, OpTypeStruct, {idStruct, idF32, idF32, idF32, idF32, idF32});
      Emit(out, OpTypePointer, {idPtrStruct, StoragePhysicalStorageBuffer, idStruct});
      Emit(out, OpTypePointer, {idPtrFloat, StoragePhysicalStorageBuffer, idF32});
      Emit(out, OpConstant, {idU64, idAddr, uint32_t(paramAddress & 0xFFFFFFFFu), uint32_t(paramAddress >> 32)});
      Emit(out, OpConstant, {idU32, idC0, 0});
      Emit(out, OpConstant, {idU32, idC1, 1});
      Emit(out, OpConstant, {idU32, idC4, 4});
    }
    if (op == OpMemoryModel && w[1] == AddrLogical) {
      Emit(out, OpMemoryModel, {AddrPhysicalStorageBuffer64, w[2]});
      continue;
    }
    if (op == OpDecorate && wc >= 3 && renamed.count(w[1])) {
      // NoContraction & co. stay on the arithmetic product
      size_t at = out.size();
      out.insert(out.end(), w, w + wc);
      out[at + 1] = renamed[w[1]];
      continue;
    }
    auto s = (ii >= firstFunc && wc >= 3) ? sites.find(w[2]) : sites.end();
    if (s == sites.end() || !types.count(w[1])) {
      out.insert(out.end(), w, w + wc);
      continue;
    }
    // the product gets a new id; the corrected value takes over the old one (all uses follow)
    uint32_t orig = w[2], p0 = renamed[orig];
    size_t at = out.size();
    out.insert(out.end(), w, w + wc);
    out[at + 2] = p0;
    uint32_t ptr = fresh(), pk = fresh(), k = fresh(), pc = fresh(), cv = fresh(), pf = fresh(), f = fresh(),
             kf = fresh();
    Emit(out, OpConvertUToPtr, {idPtrStruct, ptr, idAddr});
    Emit(out, OpAccessChain, {idPtrFloat, pk, ptr, idC0});
    Emit(out, OpLoad, {idF32, k, pk, MemAligned, 4});
    Emit(out, OpAccessChain, {idPtrFloat, pc, ptr, idC1});
    Emit(out, OpLoad, {idF32, cv, pc, MemAligned, 4});
    Emit(out, OpAccessChain, {idPtrFloat, pf, ptr, idC4});
    Emit(out, OpLoad, {idF32, f, pf, MemAligned, 4});
    Emit(out, OpFMul, {idF32, kf, k, f});
    const auto& list = s->second;
    if (!list[0]->vec) {
      uint32_t d = fresh(), sh = fresh();
      Emit(out, OpFSub, {idF32, d, list[0]->w, cv});
      Emit(out, OpFMul, {idF32, sh, kf, d});
      Emit(out, OpFSub, {idF32, orig, p0, sh});
    } else {
      uint32_t cur = p0;
      for (size_t j = 0; j < list.size(); ++j) {
        const Cand* c = list[j];
        uint32_t x = fresh(), wv = c->w, d = fresh(), sh = fresh(), nx = fresh();
        Emit(out, OpCompositeExtract, {idF32, x, p0, c->comp});
        if (c->wComp >= 0) {
          wv = fresh();
          Emit(out, OpCompositeExtract, {idF32, wv, c->w, uint32_t(c->wComp)});
        }
        Emit(out, OpFSub, {idF32, d, wv, cv});
        Emit(out, OpFMul, {idF32, sh, kf, d});
        Emit(out, OpFSub, {idF32, nx, x, sh});
        uint32_t next = j + 1 == list.size() ? orig : fresh();
        Emit(out, OpCompositeInsert, {w[1], next, nx, cur, c->comp});
        cur = next;
      }
    }
    ++patched;
  }
  out[3] = bound;
  if (log) *log = "screen-space sites=" + std::to_string(patched);
  return PatchResult::Patched;
}

}  // namespace stereo3d
