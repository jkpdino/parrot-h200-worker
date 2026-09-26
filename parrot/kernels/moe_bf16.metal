#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_bf16(
    device bfloat *output [[buffer(0)]],
    device const bfloat *input [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &epsilon [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  device const bfloat4 *input4 =
      reinterpret_cast<device const bfloat4 *>(input);
  device const bfloat4 *weight4 =
      reinterpret_cast<device const bfloat4 *>(weight);
  device bfloat4 *output4 = reinterpret_cast<device bfloat4 *>(output);
  const uint dim4 = dim / 4;
  const uint rowOffset = row * dim4;
  float squareSum = 0.0f;
  for (uint d = 0; d < dim4; ++d) {
    const float4 value = float4(input4[rowOffset + d]);
    squareSum += dot(value, value);
  }
  const float inverseRms = rsqrt(squareSum / float(dim) + epsilon);
  for (uint d = 0; d < dim4; ++d) {
    output4[rowOffset + d] = bfloat4(
        float4(input4[rowOffset + d]) * inverseRms * float4(weight4[d]));
  }
}

kernel void rms_norm_decode_bf16(
    device bfloat *output [[buffer(0)]],
    device const bfloat *input [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &epsilon [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint row = group;
  if (row >= rows) return;
  threadgroup float partial[8];
  threadgroup float inverseRms;
  float squareSum = 0.0f;
  for (uint d = tid; d < dim; d += 256) {
    const float value = float(input[row * dim + d]);
    squareSum += value * value;
  }
  squareSum = simd_sum(squareSum);
  if (lane == 0) partial[simdgroupIndex] = squareSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint index = 0; index < 8; ++index) total += partial[index];
    inverseRms = rsqrt(total / float(dim) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < dim; d += 256) {
    output[row * dim + d] = bfloat(
        float(input[row * dim + d]) * inverseRms * float(weight[d]));
  }
}

kernel void moe_route_bf16(
    device float *probabilities [[buffer(0)]],
    device float *routeWeights [[buffer(1)]],
    device long *indices [[buffer(2)]],
    device int *positions [[buffer(3)]],
    device atomic_uint *counts [[buffer(4)]],
    device const bfloat *input [[buffer(5)]],
    device const bfloat *router [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &experts [[buffer(9)]],
    constant uint &writeProbabilities [[buffer(10)]],
    uint token [[thread_position_in_grid]]) {
  if (token >= tokens) return;
  device const bfloat4 *input4 =
      reinterpret_cast<device const bfloat4 *>(input);
  device const bfloat4 *router4 =
      reinterpret_cast<device const bfloat4 *>(router);
  const uint dim4 = dim / 4;
  const uint inputOffset = token * dim4;
  float logits[32];
  float maximum = -INFINITY;
  for (uint expert = 0; expert < experts; ++expert) {
    float value = 0.0f;
    const uint weightOffset = expert * dim4;
    for (uint d = 0; d < dim4; ++d) {
      value += dot(float4(input4[inputOffset + d]),
                   float4(router4[weightOffset + d]));
    }
    logits[expert] = value;
    maximum = max(maximum, value);
  }
  float denominator = 0.0f;
  for (uint expert = 0; expert < experts; ++expert) {
    logits[expert] = fast::exp(logits[expert] - maximum);
    denominator += logits[expert];
  }
  uint first = 0;
  uint second = 1;
  if (logits[second] > logits[first]) {
    const uint swap = first;
    first = second;
    second = swap;
  }
  for (uint expert = 2; expert < experts; ++expert) {
    if (logits[expert] > logits[first]) {
      second = first;
      first = expert;
    } else if (logits[expert] > logits[second]) {
      second = expert;
    }
  }
  if (writeProbabilities != 0) {
    for (uint expert = 0; expert < experts; ++expert) {
      probabilities[token * experts + expert] = logits[expert] / denominator;
    }
  }
  const float selectedDenominator = logits[first] + logits[second];
  const uint assignment = token * 2;
  indices[assignment] = long(first);
  indices[assignment + 1] = long(second);
  routeWeights[assignment] = logits[first] / selectedDenominator;
  routeWeights[assignment + 1] = logits[second] / selectedDenominator;
  positions[assignment] = int(atomic_fetch_add_explicit(
      &counts[first], 1u, memory_order_relaxed));
  positions[assignment + 1] = int(atomic_fetch_add_explicit(
      &counts[second], 1u, memory_order_relaxed));
}

inline device const bfloat *depth_source(
    uint index,
    device const bfloat *s0, device const bfloat *s1,
    device const bfloat *s2, device const bfloat *s3,
    device const bfloat *s4, device const bfloat *s5,
    device const bfloat *s6, device const bfloat *s7,
    device const bfloat *s8) {
  switch (index) {
    case 1: return s1;
    case 2: return s2;
    case 3: return s3;
    case 4: return s4;
    case 5: return s5;
    case 6: return s6;
    case 7: return s7;
    case 8: return s8;
    default: return s0;
  }
}

inline void depth_mix_normalize_group(
    threadgroup bfloat *normalized,
    threadgroup float *squarePartials,
    threadgroup float *dotPartials,
    threadgroup float *scores,
    threadgroup float *mixWeights,
    threadgroup float *inverseOutputRms,
    device const bfloat *s0, device const bfloat *s1,
    device const bfloat *s2, device const bfloat *s3,
    device const bfloat *s4, device const bfloat *s5,
    device const bfloat *s6, device const bfloat *s7,
    device const bfloat *s8,
    device const bfloat *query,
    device const bfloat *scoreNormWeight,
    device const bfloat *outputNormWeight,
    uint token, uint dim, uint sourceCount,
    float scoreEpsilon, float outputEpsilon,
    uint tid, uint lane, uint simdgroupIndex) {
  for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
    device const bfloat *source = depth_source(
        sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
    float squareSum = 0.0f;
    float dotSum = 0.0f;
    for (uint d = tid; d < dim; d += 512) {
      const float value = float(source[token * dim + d]);
      squareSum += value * value;
      dotSum += value * float(scoreNormWeight[d]) * float(query[d]);
    }
    squareSum = simd_sum(squareSum);
    dotSum = simd_sum(dotSum);
    if (lane == 0) {
      squarePartials[simdgroupIndex] = squareSum;
      dotPartials[simdgroupIndex] = dotSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float squareTotal = 0.0f;
      float dotTotal = 0.0f;
      for (uint index = 0; index < 16; ++index) {
        squareTotal += squarePartials[index];
        dotTotal += dotPartials[index];
      }
      scores[sourceIndex] = dotTotal *
          rsqrt(squareTotal / float(dim) + scoreEpsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float maximum = -INFINITY;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      maximum = max(maximum, scores[sourceIndex]);
    }
    float denominator = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] = fast::exp(scores[sourceIndex] - maximum);
      denominator += mixWeights[sourceIndex];
    }
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] /= denominator;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float outputSquareSum = 0.0f;
  for (uint d = tid; d < dim; d += 512) {
    float value = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      device const bfloat *source = depth_source(
          sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
      value += mixWeights[sourceIndex] * float(source[token * dim + d]);
    }
    const bfloat rounded = bfloat(value);
    normalized[d] = rounded;
    const float normalizedInput = float(rounded);
    outputSquareSum += normalizedInput * normalizedInput;
  }
  outputSquareSum = simd_sum(outputSquareSum);
  if (lane == 0) squarePartials[simdgroupIndex] = outputSquareSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float squareTotal = 0.0f;
    for (uint index = 0; index < 16; ++index) {
      squareTotal += squarePartials[index];
    }
    inverseOutputRms[0] = rsqrt(squareTotal / float(dim) + outputEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < dim; d += 512) {
    normalized[d] = bfloat(
        float(normalized[d]) * inverseOutputRms[0] * float(outputNormWeight[d]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void depth_mix_decode_bf16(
    device bfloat *output [[buffer(0)]],
    device const bfloat *s0 [[buffer(1)]],
    device const bfloat *s1 [[buffer(2)]],
    device const bfloat *s2 [[buffer(3)]],
    device const bfloat *s3 [[buffer(4)]],
    device const bfloat *s4 [[buffer(5)]],
    device const bfloat *s5 [[buffer(6)]],
    device const bfloat *s6 [[buffer(7)]],
    device const bfloat *s7 [[buffer(8)]],
    device const bfloat *s8 [[buffer(9)]],
    device const bfloat *query [[buffer(10)]],
    device const bfloat *normWeight [[buffer(11)]],
    constant uint &tokens [[buffer(12)]],
    constant uint &dim [[buffer(13)]],
    constant uint &sourceCount [[buffer(14)]],
    constant float &epsilon [[buffer(15)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint token = group;
  if (token >= tokens) return;
  threadgroup float squarePartials[8];
  threadgroup float dotPartials[8];
  threadgroup float scores[9];
  threadgroup float mixWeights[9];
  for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
    device const bfloat *source = depth_source(
        sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
    float squareSum = 0.0f;
    float dotSum = 0.0f;
    for (uint d = tid; d < dim; d += 256) {
      const float value = float(source[token * dim + d]);
      squareSum += value * value;
      dotSum += value * float(normWeight[d]) * float(query[d]);
    }
    squareSum = simd_sum(squareSum);
    dotSum = simd_sum(dotSum);
    if (lane == 0) {
      squarePartials[simdgroupIndex] = squareSum;
      dotPartials[simdgroupIndex] = dotSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float squareTotal = 0.0f;
      float dotTotal = 0.0f;
      for (uint index = 0; index < 8; ++index) {
        squareTotal += squarePartials[index];
        dotTotal += dotPartials[index];
      }
      scores[sourceIndex] = dotTotal *
          rsqrt(squareTotal / float(dim) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float maximum = -INFINITY;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      maximum = max(maximum, scores[sourceIndex]);
    }
    float denominator = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] = fast::exp(scores[sourceIndex] - maximum);
      denominator += mixWeights[sourceIndex];
    }
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] /= denominator;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < dim; d += 256) {
    float value = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      device const bfloat *source = depth_source(
          sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
      value += mixWeights[sourceIndex] * float(source[token * dim + d]);
    }
    output[token * dim + d] = bfloat(value);
  }
}

kernel void depth_mix_norm_decode_bf16(
    device bfloat *output [[buffer(0)]],
    device const bfloat *s0 [[buffer(1)]],
    device const bfloat *s1 [[buffer(2)]],
    device const bfloat *s2 [[buffer(3)]],
    device const bfloat *s3 [[buffer(4)]],
    device const bfloat *s4 [[buffer(5)]],
    device const bfloat *s5 [[buffer(6)]],
    device const bfloat *s6 [[buffer(7)]],
    device const bfloat *s7 [[buffer(8)]],
    device const bfloat *s8 [[buffer(9)]],
    device const bfloat *query [[buffer(10)]],
    device const bfloat *scoreNormWeight [[buffer(11)]],
    device const bfloat *outputNormWeight [[buffer(12)]],
    constant uint &tokens [[buffer(13)]],
    constant uint &dim [[buffer(14)]],
    constant uint &sourceCount [[buffer(15)]],
    constant float &scoreEpsilon [[buffer(16)]],
    constant float &outputEpsilon [[buffer(17)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint token = group;
  if (token >= tokens) return;
  threadgroup float squarePartials[8];
  threadgroup float dotPartials[8];
  threadgroup float scores[9];
  threadgroup float mixWeights[9];
  threadgroup float inverseOutputRms;
  threadgroup bfloat mixed[768];
  for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
    device const bfloat *source = depth_source(
        sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
    float squareSum = 0.0f;
    float dotSum = 0.0f;
    for (uint d = tid; d < dim; d += 256) {
      const float value = float(source[token * dim + d]);
      squareSum += value * value;
      dotSum += value * float(scoreNormWeight[d]) * float(query[d]);
    }
    squareSum = simd_sum(squareSum);
    dotSum = simd_sum(dotSum);
    if (lane == 0) {
      squarePartials[simdgroupIndex] = squareSum;
      dotPartials[simdgroupIndex] = dotSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float squareTotal = 0.0f;
      float dotTotal = 0.0f;
      for (uint index = 0; index < 8; ++index) {
        squareTotal += squarePartials[index];
        dotTotal += dotPartials[index];
      }
      scores[sourceIndex] = dotTotal *
          rsqrt(squareTotal / float(dim) + scoreEpsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float maximum = -INFINITY;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      maximum = max(maximum, scores[sourceIndex]);
    }
    float denominator = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] = fast::exp(scores[sourceIndex] - maximum);
      denominator += mixWeights[sourceIndex];
    }
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] /= denominator;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float outputSquareSum = 0.0f;
  for (uint d = tid; d < dim; d += 256) {
    float value = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      device const bfloat *source = depth_source(
          sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
      value += mixWeights[sourceIndex] * float(source[token * dim + d]);
    }
    const bfloat rounded = bfloat(value);
    mixed[d] = rounded;
    const float normalizedInput = float(rounded);
    outputSquareSum += normalizedInput * normalizedInput;
  }
  outputSquareSum = simd_sum(outputSquareSum);
  if (lane == 0) squarePartials[simdgroupIndex] = outputSquareSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float squareTotal = 0.0f;
    for (uint index = 0; index < 8; ++index) {
      squareTotal += squarePartials[index];
    }
    inverseOutputRms = rsqrt(
        squareTotal / float(dim) + outputEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < dim; d += 256) {
    output[token * dim + d] = bfloat(
        float(mixed[d]) * inverseOutputRms * float(outputNormWeight[d]));
  }
}

kernel void depth_mix_norm_pending_bf16(
    device bfloat *output [[buffer(0)]],
    device bfloat *resolvedOutput [[buffer(1)]],
    device const float *contributions [[buffer(2)]],
    device const bfloat *base [[buffer(3)]],
    device const bfloat *s0 [[buffer(4)]],
    device const bfloat *s1 [[buffer(5)]],
    device const bfloat *s2 [[buffer(6)]],
    device const bfloat *s3 [[buffer(7)]],
    device const bfloat *s4 [[buffer(8)]],
    device const bfloat *s5 [[buffer(9)]],
    device const bfloat *s6 [[buffer(10)]],
    device const bfloat *s7 [[buffer(11)]],
    device const bfloat *s8 [[buffer(12)]],
    device const bfloat *query [[buffer(13)]],
    device const bfloat *scoreNormWeight [[buffer(14)]],
    device const bfloat *outputNormWeight [[buffer(15)]],
    constant uint &tokens [[buffer(16)]],
    constant uint &dim [[buffer(17)]],
    constant uint &sourceCount [[buffer(18)]],
    constant uint &laneCount [[buffer(19)]],
    constant uint &hasBase [[buffer(20)]],
    constant float &scoreEpsilon [[buffer(21)]],
    constant float &outputEpsilon [[buffer(22)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint token = group;
  if (token >= tokens) return;
  threadgroup bfloat resolved[768];
  threadgroup float squarePartials[8];
  threadgroup float dotPartials[8];
  threadgroup float scores[9];
  threadgroup float mixWeights[9];
  threadgroup float inverseOutputRms;
  for (uint d = tid; d < dim; d += 256) {
    float value = 0.0f;
    for (uint contribution = 0; contribution < laneCount; ++contribution) {
      value += contributions[(token * laneCount + contribution) * dim + d];
    }
    const bfloat delta = bfloat(value);
    const bfloat combined = hasBase != 0
        ? bfloat(float(base[token * dim + d]) + float(delta)) : delta;
    resolved[d] = combined;
    resolvedOutput[token * dim + d] = combined;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint pendingIndex = sourceCount - 1;
  for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
    device const bfloat *source = depth_source(
        sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
    float squareSum = 0.0f;
    float dotSum = 0.0f;
    for (uint d = tid; d < dim; d += 256) {
      const float value = sourceIndex == pendingIndex
          ? float(resolved[d]) : float(source[token * dim + d]);
      squareSum += value * value;
      dotSum += value * float(scoreNormWeight[d]) * float(query[d]);
    }
    squareSum = simd_sum(squareSum);
    dotSum = simd_sum(dotSum);
    if (lane == 0) {
      squarePartials[simdgroupIndex] = squareSum;
      dotPartials[simdgroupIndex] = dotSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float squareTotal = 0.0f;
      float dotTotal = 0.0f;
      for (uint index = 0; index < 8; ++index) {
        squareTotal += squarePartials[index];
        dotTotal += dotPartials[index];
      }
      scores[sourceIndex] = dotTotal *
          rsqrt(squareTotal / float(dim) + scoreEpsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    float maximum = -INFINITY;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      maximum = max(maximum, scores[sourceIndex]);
    }
    float denominator = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] = fast::exp(scores[sourceIndex] - maximum);
      denominator += mixWeights[sourceIndex];
    }
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      mixWeights[sourceIndex] /= denominator;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float outputSquareSum = 0.0f;
  for (uint d = tid; d < dim; d += 256) {
    float value = 0.0f;
    for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
      device const bfloat *source = depth_source(
          sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
      const float sourceValue = sourceIndex == pendingIndex
          ? float(resolved[d]) : float(source[token * dim + d]);
      value += mixWeights[sourceIndex] * sourceValue;
    }
    const bfloat rounded = bfloat(value);
    resolved[d] = rounded;
    const float normalizedInput = float(rounded);
    outputSquareSum += normalizedInput * normalizedInput;
  }
  outputSquareSum = simd_sum(outputSquareSum);
  if (lane == 0) squarePartials[simdgroupIndex] = outputSquareSum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float squareTotal = 0.0f;
    for (uint index = 0; index < 8; ++index) {
      squareTotal += squarePartials[index];
    }
    inverseOutputRms = rsqrt(squareTotal / float(dim) + outputEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = tid; d < dim; d += 256) {
    output[token * dim + d] = bfloat(
        float(resolved[d]) * inverseOutputRms * float(outputNormWeight[d]));
  }
}

kernel void depth_scores_bf16(
    device float *scores [[buffer(0)]],
    device const bfloat *s0 [[buffer(1)]],
    device const bfloat *s1 [[buffer(2)]],
    device const bfloat *s2 [[buffer(3)]],
    device const bfloat *s3 [[buffer(4)]],
    device const bfloat *s4 [[buffer(5)]],
    device const bfloat *s5 [[buffer(6)]],
    device const bfloat *s6 [[buffer(7)]],
    device const bfloat *s7 [[buffer(8)]],
    device const bfloat *s8 [[buffer(9)]],
    device const bfloat *query [[buffer(10)]],
    device const bfloat *normWeight [[buffer(11)]],
    constant uint &tokens [[buffer(12)]],
    constant uint &dim [[buffer(13)]],
    constant uint &sourceCount [[buffer(14)]],
    constant float &epsilon [[buffer(15)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = sourceCount * tokens;
  if (gid >= count) return;
  const uint sourceIndex = gid / tokens;
  const uint token = gid - sourceIndex * tokens;
  device const bfloat *source = depth_source(
      sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
  device const bfloat4 *source4 =
      reinterpret_cast<device const bfloat4 *>(source);
  device const bfloat4 *query4 =
      reinterpret_cast<device const bfloat4 *>(query);
  device const bfloat4 *weight4 =
      reinterpret_cast<device const bfloat4 *>(normWeight);
  const uint offset4 = token * dim / 4;
  const uint dim4 = dim / 4;
  float squareSum = 0.0f;
  for (uint d = 0; d < dim4; ++d) {
    const float4 value = float4(source4[offset4 + d]);
    squareSum += dot(value, value);
  }
  const float inverseRms = rsqrt(squareSum / float(dim) + epsilon);
  float score = 0.0f;
  for (uint d = 0; d < dim4; ++d) {
    const float4 value = float4(source4[offset4 + d]);
    score += dot(value * inverseRms * float4(weight4[d]), float4(query4[d]));
  }
  scores[gid] = score;
}

kernel void depth_mix_bf16(
    device bfloat *output [[buffer(0)]],
    device const float *scores [[buffer(1)]],
    device const bfloat *s0 [[buffer(2)]],
    device const bfloat *s1 [[buffer(3)]],
    device const bfloat *s2 [[buffer(4)]],
    device const bfloat *s3 [[buffer(5)]],
    device const bfloat *s4 [[buffer(6)]],
    device const bfloat *s5 [[buffer(7)]],
    device const bfloat *s6 [[buffer(8)]],
    device const bfloat *s7 [[buffer(9)]],
    device const bfloat *s8 [[buffer(10)]],
    constant uint &tokens [[buffer(11)]],
    constant uint &dim [[buffer(12)]],
    constant uint &sourceCount [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = tokens * dim;
  if (gid >= count) return;
  const uint token = gid / dim;
  float maximum = -INFINITY;
  for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
    maximum = max(maximum, scores[sourceIndex * tokens + token]);
  }
  float denominator = 0.0f;
  for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
    denominator += fast::exp(scores[sourceIndex * tokens + token] - maximum);
  }
  float value = 0.0f;
  for (uint sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
    device const bfloat *source = depth_source(
        sourceIndex, s0, s1, s2, s3, s4, s5, s6, s7, s8);
    const float weight = fast::exp(scores[sourceIndex * tokens + token] - maximum) /
                         denominator;
    value += weight * float(source[gid]);
  }
  output[gid] = bfloat(value);
}

// PyTorch MPS inference kernel. The router and top-k stay in PyTorch; these
// kernels replace the many per-expert gather/matmul/scatter launches.

kernel void moe_count_assignments(
    device int *positions [[buffer(0)]],
    device atomic_uint *counts [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    constant uint &assignments [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= assignments) return;
  const uint expert = uint(indices[gid]);
  positions[gid] = int(atomic_fetch_add_explicit(
      &counts[expert], 1u, memory_order_relaxed));
}

kernel void swiglu_from_gate_up_bf16(
    device bfloat *hidden [[buffer(0)]],
    device const bfloat *gateUp [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &hiddenDim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = rows * hiddenDim;
  if (gid >= count) return;
  const uint row = gid / hiddenDim;
  const uint h = gid - row * hiddenDim;
  const uint offset = row * hiddenDim * 2;
  const float gate = float(gateUp[offset + h]);
  const float up = float(gateUp[offset + hiddenDim + h]);
  hidden[gid] = bfloat((gate / (1.0f + fast::exp(-gate))) * up);
}

kernel void moe_scatter_grouped_bf16(
    device bfloat *grouped [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const int *positions [[buffer(3)]],
    constant uint &assignments [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &capacity [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = assignments * dim;
  if (gid >= count) return;
  const uint assignment = gid / dim;
  const uint d = gid - assignment * dim;
  const uint token = assignment / 2;
  const uint expert = uint(indices[assignment]);
  const uint slot = uint(positions[assignment]);
  if (slot >= capacity) return;
  grouped[(expert * capacity + slot) * dim + d] = x[token * dim + d];
}

kernel void moe_overflow_hidden_bf16(
    device bfloat *overflowHidden [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const int *positions [[buffer(3)]],
    device const bfloat *routedGateUp [[buffer(4)]],
    constant uint &assignments [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &expertDim [[buffer(7)]],
    constant uint &capacity [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = assignments * expertDim;
  if (gid >= count) return;
  const uint h = gid % expertDim;
  const uint assignment = gid / expertDim;
  if (uint(positions[assignment]) < capacity) return;
  const uint token = assignment / 2;
  const uint expert = uint(indices[assignment]);
  const uint inputOffset4 = token * dim / 4;
  const uint gateOffset4 = (expert * expertDim * 2 + h) * dim / 4;
  const uint upOffset4 = (expert * expertDim * 2 + expertDim + h) * dim / 4;
  const uint dim4 = dim / 4;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *gateUp4 =
      reinterpret_cast<device const bfloat4 *>(routedGateUp);
  float gateValue = 0.0f;
  float upValue = 0.0f;
  for (uint d = 0; d < dim4; ++d) {
    const float4 input = float4(x4[inputOffset4 + d]);
    gateValue += dot(input, float4(gateUp4[gateOffset4 + d]));
    upValue += dot(input, float4(gateUp4[upOffset4 + d]));
  }
  overflowHidden[gid] = bfloat(
      (gateValue / (1.0f + fast::exp(-gateValue))) * upValue);
}

kernel void moe_gather_grouped_bf16(
    device bfloat *output [[buffer(0)]],
    device const bfloat *expertOutput [[buffer(1)]],
    device const bfloat *overflowHidden [[buffer(2)]],
    device const bfloat *sharedOutput [[buffer(3)]],
    device const long *indices [[buffer(4)]],
    device const int *positions [[buffer(5)]],
    device const float *routeWeights [[buffer(6)]],
    device const bfloat *routedDown [[buffer(7)]],
    constant uint &tokens [[buffer(8)]],
    constant uint &dim [[buffer(9)]],
    constant uint &expertDim [[buffer(10)]],
    constant uint &capacity [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = tokens * dim;
  if (gid >= count) return;
  const uint token = gid / dim;
  const uint d = gid - token * dim;
  float value = float(sharedOutput[gid]);
  for (uint lane = 0; lane < 2; ++lane) {
    const uint assignment = token * 2 + lane;
    const uint expert = uint(indices[assignment]);
    const uint slot = uint(positions[assignment]);
    float expertValue;
    if (slot < capacity) {
      expertValue = float(expertOutput[(expert * capacity + slot) * dim + d]);
    } else {
      const uint hiddenOffset4 = assignment * expertDim / 4;
      const uint weightOffset4 = (expert * dim + d) * expertDim / 4;
      const uint expertDim4 = expertDim / 4;
      device const bfloat4 *hidden4 =
          reinterpret_cast<device const bfloat4 *>(overflowHidden);
      device const bfloat4 *down4 =
          reinterpret_cast<device const bfloat4 *>(routedDown);
      expertValue = 0.0f;
      for (uint h = 0; h < expertDim4; ++h) {
        expertValue += dot(float4(hidden4[hiddenOffset4 + h]),
                           float4(down4[weightOffset4 + h]));
      }
    }
    value += routeWeights[assignment] * expertValue;
  }
  output[gid] = bfloat(value);
}

kernel void moe_swiglu_hidden_bf16(
    device bfloat *hidden [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const bfloat *routedGate [[buffer(3)]],
    device const bfloat *routedUp [[buffer(4)]],
    device const bfloat *sharedGate [[buffer(5)]],
    device const bfloat *sharedUp [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &expertDim [[buffer(9)]],
    constant uint &sharedDim [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  const uint lanes = 3;
  const uint count = tokens * lanes * expertDim;
  if (gid >= count) return;
  const uint h = gid % expertDim;
  const uint laneAndToken = gid / expertDim;
  const uint lane = laneAndToken % lanes;
  const uint token = laneAndToken / lanes;
  const bool shared = lane == 2;
  if (shared && h >= sharedDim) {
    hidden[gid] = bfloat(0.0f);
    return;
  }

  const uint inputOffset = token * dim;
  uint weightOffset;
  float gateValue = 0.0f;
  float upValue = 0.0f;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *routedGate4 =
      reinterpret_cast<device const bfloat4 *>(routedGate);
  device const bfloat4 *routedUp4 =
      reinterpret_cast<device const bfloat4 *>(routedUp);
  device const bfloat4 *sharedGate4 =
      reinterpret_cast<device const bfloat4 *>(sharedGate);
  device const bfloat4 *sharedUp4 =
      reinterpret_cast<device const bfloat4 *>(sharedUp);
  const uint inputOffset4 = inputOffset / 4;
  const uint dim4 = dim / 4;
  if (shared) {
    weightOffset = h * dim;
    const uint weightOffset4 = weightOffset / 4;
    for (uint d = 0; d < dim4; ++d) {
      const float4 input = float4(x4[inputOffset4 + d]);
      gateValue += dot(input, float4(sharedGate4[weightOffset4 + d]));
      upValue += dot(input, float4(sharedUp4[weightOffset4 + d]));
    }
  } else {
    const uint expert = uint(indices[token * 2 + lane]);
    weightOffset = (expert * expertDim + h) * dim;
    const uint weightOffset4 = weightOffset / 4;
    for (uint d = 0; d < dim4; ++d) {
      const float4 input = float4(x4[inputOffset4 + d]);
      gateValue += dot(input, float4(routedGate4[weightOffset4 + d]));
      upValue += dot(input, float4(routedUp4[weightOffset4 + d]));
    }
  }
  const float silu = gateValue / (1.0f + fast::exp(-gateValue));
  hidden[gid] = bfloat(silu * upValue);
}

kernel void moe_down_accumulate_bf16(
    device bfloat *output [[buffer(0)]],
    device const bfloat *hidden [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const float *routeWeights [[buffer(3)]],
    device const bfloat *routedDown [[buffer(4)]],
    device const bfloat *sharedDown [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant uint &expertDim [[buffer(8)]],
    constant uint &sharedDim [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = tokens * dim;
  if (gid >= count) return;
  const uint token = gid / dim;
  const uint out = gid - token * dim;
  float sum = 0.0f;
  device const bfloat4 *hidden4 =
      reinterpret_cast<device const bfloat4 *>(hidden);
  device const bfloat4 *routedDown4 =
      reinterpret_cast<device const bfloat4 *>(routedDown);
  device const bfloat4 *sharedDown4 =
      reinterpret_cast<device const bfloat4 *>(sharedDown);
  const uint expertDim4 = expertDim / 4;

  for (uint lane = 0; lane < 2; ++lane) {
    const uint expert = uint(indices[token * 2 + lane]);
    const uint hiddenOffset = (token * 3 + lane) * expertDim;
    const uint weightOffset = (expert * dim + out) * expertDim;
    float projected = 0.0f;
    const uint hiddenOffset4 = hiddenOffset / 4;
    const uint weightOffset4 = weightOffset / 4;
    for (uint h = 0; h < expertDim4; ++h) {
      projected += dot(float4(hidden4[hiddenOffset4 + h]),
                       float4(routedDown4[weightOffset4 + h]));
    }
    sum += routeWeights[token * 2 + lane] * projected;
  }

  const uint sharedHiddenOffset = (token * 3 + 2) * expertDim;
  const uint sharedWeightOffset = out * sharedDim;
  float sharedValue = 0.0f;
  const uint sharedHiddenOffset4 = sharedHiddenOffset / 4;
  const uint sharedWeightOffset4 = sharedWeightOffset / 4;
  const uint sharedDim4 = sharedDim / 4;
  for (uint h = 0; h < sharedDim4; ++h) {
    sharedValue += dot(float4(hidden4[sharedHiddenOffset4 + h]),
                       float4(sharedDown4[sharedWeightOffset4 + h]));
  }
  output[gid] = bfloat(sum + sharedValue);
}

// Small-batch decode path. Each token launches three threadgroups in parallel:
// top-1, top-2, and the shared expert. Routed groups independently compute the
// tiny router so no routing dispatch or cross-threadgroup synchronization is
// needed. Expert outputs stay FP32 until the following sum kernel.
kernel void moe_decode_lanes_bf16(
    device float *laneOutput [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device const bfloat *router [[buffer(2)]],
    device const bfloat *routedGateUp [[buffer(3)]],
    device const bfloat *routedDown [[buffer(4)]],
    device const bfloat *sharedGateUp [[buffer(5)]],
    device const bfloat *sharedDown [[buffer(6)]],
    constant uint &tokens [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &experts [[buffer(9)]],
    constant uint &expertDim [[buffer(10)]],
    constant uint &sharedDim [[buffer(11)]],
    uint tid [[thread_position_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint token = group / 3;
  const uint expertLane = group - token * 3;
  if (token >= tokens) return;

  threadgroup float logits[32];
  threadgroup uint selected[2];
  threadgroup float routeWeights[2];
  threadgroup bfloat hidden[512];
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *router4 =
      reinterpret_cast<device const bfloat4 *>(router);
  device const bfloat4 *routedGateUp4 =
      reinterpret_cast<device const bfloat4 *>(routedGateUp);
  device const bfloat4 *routedDown4 =
      reinterpret_cast<device const bfloat4 *>(routedDown);
  device const bfloat4 *sharedGateUp4 =
      reinterpret_cast<device const bfloat4 *>(sharedGateUp);
  device const bfloat4 *sharedDown4 =
      reinterpret_cast<device const bfloat4 *>(sharedDown);
  threadgroup bfloat4 *hidden4 =
      reinterpret_cast<threadgroup bfloat4 *>(hidden);
  const uint dim4 = dim / 4;
  const uint inputOffset4 = token * dim4;

  if (expertLane < 2) {
    if (tid < experts) {
      float value = 0.0f;
      const uint weightOffset4 = tid * dim4;
      for (uint d = 0; d < dim4; ++d) {
        value += dot(float4(x4[inputOffset4 + d]),
                     float4(router4[weightOffset4 + d]));
      }
      logits[tid] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint first = 0;
      uint second = 1;
      if (logits[second] > logits[first]) {
        const uint swap = first;
        first = second;
        second = swap;
      }
      for (uint expert = 2; expert < experts; ++expert) {
        if (logits[expert] > logits[first]) {
          second = first;
          first = expert;
        } else if (logits[expert] > logits[second]) {
          second = expert;
        }
      }
      selected[0] = first;
      selected[1] = second;
      const float maximum = max(logits[first], logits[second]);
      const float firstWeight = fast::exp(logits[first] - maximum);
      const float secondWeight = fast::exp(logits[second] - maximum);
      const float denominator = firstWeight + secondWeight;
      routeWeights[0] = firstWeight / denominator;
      routeWeights[1] = secondWeight / denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint hiddenDim = expertLane < 2 ? expertDim : sharedDim;
  for (uint h = tid; h < hiddenDim; h += groupSize) {
    float gateValue = 0.0f;
    float upValue = 0.0f;
    if (expertLane < 2) {
      const uint expert = selected[expertLane];
      const uint weightOffset4 = (expert * 2 * expertDim + h) * dim4;
      const uint upOffset4 = weightOffset4 + expertDim * dim4;
      for (uint d = 0; d < dim4; ++d) {
        const float4 input = float4(x4[inputOffset4 + d]);
        gateValue += dot(input, float4(routedGateUp4[weightOffset4 + d]));
        upValue += dot(input, float4(routedGateUp4[upOffset4 + d]));
      }
    } else {
      const uint weightOffset4 = h * dim4;
      const uint upOffset4 = weightOffset4 + sharedDim * dim4;
      for (uint d = 0; d < dim4; ++d) {
        const float4 input = float4(x4[inputOffset4 + d]);
        gateValue += dot(input, float4(sharedGateUp4[weightOffset4 + d]));
        upValue += dot(input, float4(sharedGateUp4[upOffset4 + d]));
      }
    }
    const float silu = gateValue / (1.0f + fast::exp(-gateValue));
    hidden[h] = bfloat(silu * upValue);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint hiddenDim4 = hiddenDim / 4;
  for (uint out = tid; out < dim; out += groupSize) {
    float value = 0.0f;
    if (expertLane < 2) {
      const uint expert = selected[expertLane];
      const uint weightOffset4 = (expert * dim + out) * (expertDim / 4);
      for (uint h = 0; h < hiddenDim4; ++h) {
        value += dot(float4(hidden4[h]),
                     float4(routedDown4[weightOffset4 + h]));
      }
      value *= routeWeights[expertLane];
    } else {
      const uint weightOffset4 = out * (sharedDim / 4);
      for (uint h = 0; h < hiddenDim4; ++h) {
        value += dot(float4(hidden4[h]),
                     float4(sharedDown4[weightOffset4 + h]));
      }
    }
    laneOutput[(token * 3 + expertLane) * dim + out] = value;
  }
}

kernel void moe_depth_decode_lanes_bf16(
    device float *laneOutput [[buffer(0)]],
    device const bfloat *router [[buffer(1)]],
    device const bfloat *routedGate [[buffer(2)]],
    device const bfloat *routedUp [[buffer(3)]],
    device const bfloat *routedDown [[buffer(4)]],
    device const bfloat *sharedGate [[buffer(5)]],
    device const bfloat *sharedUp [[buffer(6)]],
    device const bfloat *sharedDown [[buffer(7)]],
    device const bfloat *s0 [[buffer(8)]],
    device const bfloat *s1 [[buffer(9)]],
    device const bfloat *s2 [[buffer(10)]],
    device const bfloat *s3 [[buffer(11)]],
    device const bfloat *s4 [[buffer(12)]],
    device const bfloat *s5 [[buffer(13)]],
    device const bfloat *s6 [[buffer(14)]],
    device const bfloat *s7 [[buffer(15)]],
    device const bfloat *s8 [[buffer(16)]],
    device const bfloat *depthQuery [[buffer(17)]],
    device const bfloat *scoreNormWeight [[buffer(18)]],
    device const bfloat *outputNormWeight [[buffer(19)]],
    constant uint &tokens [[buffer(20)]],
    constant uint &dim [[buffer(21)]],
    constant uint &experts [[buffer(22)]],
    constant uint &expertDim [[buffer(23)]],
    constant uint &sharedDim [[buffer(24)]],
    constant uint &sourceCount [[buffer(25)]],
    constant float &scoreEpsilon [[buffer(26)]],
    constant float &outputEpsilon [[buffer(27)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]]) {
  const uint token = group / 3;
  const uint expertLane = group - token * 3;
  if (token >= tokens) return;
  threadgroup bfloat normalized[768];
  threadgroup float squarePartials[16];
  threadgroup float dotPartials[16];
  threadgroup float depthScores[9];
  threadgroup float mixWeights[9];
  threadgroup float inverseDepthRms;
  depth_mix_normalize_group(
      normalized, squarePartials, dotPartials, depthScores, mixWeights,
      &inverseDepthRms,
      s0, s1, s2, s3, s4, s5, s6, s7, s8,
      depthQuery, scoreNormWeight, outputNormWeight,
      token, dim, sourceCount, scoreEpsilon, outputEpsilon,
      tid, lane, simdgroupIndex);

  threadgroup float logits[32];
  threadgroup uint selected[2];
  threadgroup float routeWeights[2];
  threadgroup bfloat hidden[512];
  threadgroup const bfloat4 *x4 =
      reinterpret_cast<threadgroup bfloat4 *>(normalized);
  device const bfloat4 *router4 =
      reinterpret_cast<device const bfloat4 *>(router);
  device const bfloat4 *routedGate4 =
      reinterpret_cast<device const bfloat4 *>(routedGate);
  device const bfloat4 *routedUp4 =
      reinterpret_cast<device const bfloat4 *>(routedUp);
  device const bfloat4 *routedDown4 =
      reinterpret_cast<device const bfloat4 *>(routedDown);
  device const bfloat4 *sharedGate4 =
      reinterpret_cast<device const bfloat4 *>(sharedGate);
  device const bfloat4 *sharedUp4 =
      reinterpret_cast<device const bfloat4 *>(sharedUp);
  device const bfloat4 *sharedDown4 =
      reinterpret_cast<device const bfloat4 *>(sharedDown);
  threadgroup bfloat4 *hidden4 =
      reinterpret_cast<threadgroup bfloat4 *>(hidden);
  const uint dim4 = dim / 4;

  if (expertLane < 2) {
    if (tid < experts) {
      float value = 0.0f;
      const uint weightOffset4 = tid * dim4;
      for (uint d = 0; d < dim4; ++d) {
        value += dot(float4(x4[d]), float4(router4[weightOffset4 + d]));
      }
      logits[tid] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint first = 0;
      uint second = 1;
      if (logits[second] > logits[first]) {
        const uint swap = first;
        first = second;
        second = swap;
      }
      for (uint expert = 2; expert < experts; ++expert) {
        if (logits[expert] > logits[first]) {
          second = first;
          first = expert;
        } else if (logits[expert] > logits[second]) {
          second = expert;
        }
      }
      selected[0] = first;
      selected[1] = second;
      const float maximum = max(logits[first], logits[second]);
      const float firstWeight = fast::exp(logits[first] - maximum);
      const float secondWeight = fast::exp(logits[second] - maximum);
      const float denominator = firstWeight + secondWeight;
      routeWeights[0] = firstWeight / denominator;
      routeWeights[1] = secondWeight / denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint hiddenDim = expertLane < 2 ? expertDim : sharedDim;
  if (tid < hiddenDim) {
    float gateValue = 0.0f;
    float upValue = 0.0f;
    if (expertLane < 2) {
      const uint expert = selected[expertLane];
      const uint weightOffset4 = (expert * expertDim + tid) * dim4;
      for (uint d = 0; d < dim4; ++d) {
        const float4 input = float4(x4[d]);
        gateValue += dot(input, float4(routedGate4[weightOffset4 + d]));
        upValue += dot(input, float4(routedUp4[weightOffset4 + d]));
      }
    } else {
      const uint weightOffset4 = tid * dim4;
      for (uint d = 0; d < dim4; ++d) {
        const float4 input = float4(x4[d]);
        gateValue += dot(input, float4(sharedGate4[weightOffset4 + d]));
        upValue += dot(input, float4(sharedUp4[weightOffset4 + d]));
      }
    }
    const float silu = gateValue / (1.0f + fast::exp(-gateValue));
    hidden[tid] = bfloat(silu * upValue);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint hiddenDim4 = hiddenDim / 4;
  for (uint out = tid; out < dim; out += 512) {
    float value = 0.0f;
    if (expertLane < 2) {
      const uint expert = selected[expertLane];
      const uint weightOffset4 = (expert * dim + out) * (expertDim / 4);
      for (uint h = 0; h < hiddenDim4; ++h) {
        value += dot(float4(hidden4[h]),
                     float4(routedDown4[weightOffset4 + h]));
      }
      value *= routeWeights[expertLane];
    } else {
      const uint weightOffset4 = out * (sharedDim / 4);
      for (uint h = 0; h < hiddenDim4; ++h) {
        value += dot(float4(hidden4[h]),
                     float4(sharedDown4[weightOffset4 + h]));
      }
    }
    laneOutput[(token * 3 + expertLane) * dim + out] = value;
  }
}

kernel void moe_sum_decode_lanes_bf16(
    device bfloat *output [[buffer(0)]],
    device const float *laneOutput [[buffer(1)]],
    device const bfloat *addend [[buffer(2)]],
    constant uint &tokens [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant uint &addResidual [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= tokens * dim) return;
  const uint token = gid / dim;
  const uint out = gid - token * dim;
  const uint base = token * 3 * dim + out;
  const bfloat delta = bfloat(
      laneOutput[base] + laneOutput[base + dim] + laneOutput[base + 2 * dim]);
  output[gid] = addResidual != 0
      ? bfloat(float(addend[gid]) + float(delta)) : delta;
}

// One query head per threadgroup. K/V projection is intentionally duplicated
// across the four query heads that share a KV head; this keeps all eight GPU
// cores busy and removes every intermediate dispatch before the output GEMV.
kernel void local_attention_decode_bf16(
    device float *headOutput [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device const bfloat *qWeight [[buffer(2)]],
    device const bfloat *kWeight [[buffer(3)]],
    device const bfloat *vWeight [[buffer(4)]],
    device bfloat *keyCache [[buffer(5)]],
    device bfloat *valueCache [[buffer(6)]],
    device const bfloat *ropeCos [[buffer(7)]],
    device const bfloat *ropeSin [[buffer(8)]],
    constant uint &batchSize [[buffer(9)]],
    constant uint &position [[buffer(10)]],
    constant uint &windowStart [[buffer(11)]],
    constant uint &dim [[buffer(12)]],
    constant uint &heads [[buffer(13)]],
    constant uint &kvHeads [[buffer(14)]],
    constant uint &headDim [[buffer(15)]],
    constant uint &maxSequence [[buffer(16)]],
    constant float &attentionScale [[buffer(17)]],
    device const bfloat *outputWeight [[buffer(18)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint batch = group / heads;
  const uint head = group - batch * heads;
  if (batch >= batchSize) return;
  const uint headsPerKv = heads / kvHeads;
  const uint kvHead = head / headsPerKv;
  const uint scoreCount = position - windowStart + 1;
  threadgroup bfloat q[96];
  threadgroup bfloat k[96];
  threadgroup bfloat currentValue[96];
  threadgroup bfloat headContext[96];
  threadgroup float scores[2048];
  threadgroup float reduction[8];
  threadgroup float maximum;
  threadgroup float denominator;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *qWeight4 =
      reinterpret_cast<device const bfloat4 *>(qWeight);
  device const bfloat4 *kWeight4 =
      reinterpret_cast<device const bfloat4 *>(kWeight);
  device const bfloat4 *vWeight4 =
      reinterpret_cast<device const bfloat4 *>(vWeight);
  const uint dim4 = dim / 4;
  const uint inputOffset4 = batch * dim4;

  if (tid < headDim) {
    float qValue = 0.0f;
    float kValue = 0.0f;
    float vValue = 0.0f;
    const uint qOffset4 = (head * headDim + tid) * dim4;
    const uint kvOffset4 = (kvHead * headDim + tid) * dim4;
    for (uint d = 0; d < dim4; ++d) {
      const float4 input = float4(x4[inputOffset4 + d]);
      qValue += dot(input, float4(qWeight4[qOffset4 + d]));
      kValue += dot(input, float4(kWeight4[kvOffset4 + d]));
      vValue += dot(input, float4(vWeight4[kvOffset4 + d]));
    }
    q[tid] = bfloat(qValue);
    k[tid] = bfloat(kValue);
    currentValue[tid] = bfloat(vValue);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < headDim / 2) {
    const uint even = 2 * tid;
    const uint odd = even + 1;
    const float cosine = float(ropeCos[position * (headDim / 2) + tid]);
    const float sine = float(ropeSin[position * (headDim / 2) + tid]);
    const float qEven = float(q[even]);
    const float qOdd = float(q[odd]);
    const float kEven = float(k[even]);
    const float kOdd = float(k[odd]);
    q[even] = bfloat(qEven * cosine - qOdd * sine);
    q[odd] = bfloat(qEven * sine + qOdd * cosine);
    k[even] = bfloat(kEven * cosine - kOdd * sine);
    k[odd] = bfloat(kEven * sine + kOdd * cosine);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (head % headsPerKv == 0 && tid < headDim) {
    const uint cacheOffset =
        ((batch * kvHeads + kvHead) * maxSequence + position) * headDim + tid;
    keyCache[cacheOffset] = k[tid];
    valueCache[cacheOffset] = currentValue[tid];
  }

  float localMaximum = -INFINITY;
  for (uint index = tid; index < scoreCount; index += 256) {
    const uint sequencePosition = windowStart + index;
    float score = 0.0f;
    for (uint d = 0; d < headDim; ++d) {
      const uint cacheOffset =
          ((batch * kvHeads + kvHead) * maxSequence + sequencePosition) *
          headDim + d;
      const float keyValue = sequencePosition == position
          ? float(k[d]) : float(keyCache[cacheOffset]);
      score += float(q[d]) * keyValue;
    }
    score *= attentionScale;
    scores[index] = score;
    localMaximum = max(localMaximum, score);
  }
  localMaximum = simd_max(localMaximum);
  if (lane == 0) reduction[simdgroupIndex] = localMaximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float value = -INFINITY;
    for (uint index = 0; index < 8; ++index) value = max(value, reduction[index]);
    maximum = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float localDenominator = 0.0f;
  for (uint index = tid; index < scoreCount; index += 256) {
    const float value = fast::exp(scores[index] - maximum);
    scores[index] = value;
    localDenominator += value;
  }
  localDenominator = simd_sum(localDenominator);
  if (lane == 0) reduction[simdgroupIndex] = localDenominator;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float value = 0.0f;
    for (uint index = 0; index < 8; ++index) value += reduction[index];
    denominator = 1.0f / value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < headDim) {
    float value = 0.0f;
    for (uint index = 0; index < scoreCount; ++index) {
      const uint sequencePosition = windowStart + index;
      const uint cacheOffset =
          ((batch * kvHeads + kvHead) * maxSequence + sequencePosition) *
          headDim + tid;
      const float cachedValue = sequencePosition == position
          ? float(currentValue[tid]) : float(valueCache[cacheOffset]);
      value += (scores[index] * denominator) * cachedValue;
    }
    headContext[tid] = bfloat(value);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const bfloat4 *headContext4 =
      reinterpret_cast<threadgroup bfloat4 *>(headContext);
  device const bfloat4 *outputWeight4 =
      reinterpret_cast<device const bfloat4 *>(outputWeight);
  const uint headDim4 = headDim / 4;
  for (uint out = tid; out < dim; out += 256) {
    float value = 0.0f;
    const uint weightOffset4 = (out * dim + head * headDim) / 4;
    for (uint d = 0; d < headDim4; ++d) {
      value += dot(float4(headContext4[d]),
                   float4(outputWeight4[weightOffset4 + d]));
    }
    headOutput[(batch * heads + head) * dim + out] = value;
  }
}

kernel void attention_output_decode_bf16(
    device bfloat *output [[buffer(0)]],
    device const bfloat *context [[buffer(1)]],
    device const bfloat *weight [[buffer(2)]],
    device const bfloat *addend [[buffer(3)]],
    constant uint &tokens [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &addResidual [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= tokens * dim) return;
  const uint token = gid / dim;
  const uint out = gid - token * dim;
  device const bfloat4 *context4 =
      reinterpret_cast<device const bfloat4 *>(context);
  device const bfloat4 *weight4 =
      reinterpret_cast<device const bfloat4 *>(weight);
  const uint dim4 = dim / 4;
  float value = 0.0f;
  for (uint d = 0; d < dim4; ++d) {
    value += dot(float4(context4[token * dim4 + d]),
                 float4(weight4[out * dim4 + d]));
  }
  const bfloat delta = bfloat(value);
  output[gid] = addResidual != 0
      ? bfloat(float(addend[gid]) + float(delta)) : delta;
}

kernel void mla_attention_decode_bf16(
    device float *headOutput [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device const bfloat *kvDownWeight [[buffer(2)]],
    device const bfloat *queryWeight [[buffer(3)]],
    device bfloat *latentCache [[buffer(4)]],
    constant uint &batchSize [[buffer(5)]],
    constant uint &position [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant uint &heads [[buffer(8)]],
    constant uint &latentDim [[buffer(9)]],
    constant uint &maxSequence [[buffer(10)]],
    constant float &attentionScale [[buffer(11)]],
    device const bfloat *outputWeight [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint batch = group / heads;
  const uint head = group - batch * heads;
  if (batch >= batchSize) return;
  const uint scoreCount = position + 1;
  threadgroup float currentLatent[128];
  threadgroup float query[128];
  threadgroup bfloat headContext[128];
  threadgroup float scores[4096];
  threadgroup float reduction[8];
  threadgroup float maximum;
  threadgroup float denominator;
  device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(x);
  device const bfloat4 *kvWeight4 =
      reinterpret_cast<device const bfloat4 *>(kvDownWeight);
  device const bfloat4 *queryWeight4 =
      reinterpret_cast<device const bfloat4 *>(queryWeight);
  const uint dim4 = dim / 4;
  const uint inputOffset4 = batch * dim4;

  if (tid < latentDim) {
    float latentValue = 0.0f;
    const uint kvOffset4 = tid * dim4;
    for (uint d = 0; d < dim4; ++d) {
      latentValue += dot(float4(x4[inputOffset4 + d]),
                         float4(kvWeight4[kvOffset4 + d]));
    }
    currentLatent[tid] = float(bfloat(latentValue));
    float queryValue = 0.0f;
    const uint queryOffset4 = (head * latentDim + tid) * dim4;
    for (uint d = 0; d < dim4; ++d) {
      queryValue += dot(float4(x4[inputOffset4 + d]),
                        float4(queryWeight4[queryOffset4 + d]));
    }
    query[tid] = queryValue;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (head == 0 && tid < latentDim) {
    latentCache[(batch * maxSequence + position) * latentDim + tid] =
        bfloat(currentLatent[tid]);
  }

  float localMaximum = -INFINITY;
  for (uint sequencePosition = tid; sequencePosition < scoreCount;
       sequencePosition += 256) {
    float score = 0.0f;
    for (uint d = 0; d < latentDim; ++d) {
      const float latentValue = sequencePosition == position
          ? currentLatent[d]
          : float(latentCache[
                (batch * maxSequence + sequencePosition) * latentDim + d]);
      score += query[d] * latentValue;
    }
    score *= attentionScale;
    scores[sequencePosition] = score;
    localMaximum = max(localMaximum, score);
  }
  localMaximum = simd_max(localMaximum);
  if (lane == 0) reduction[simdgroupIndex] = localMaximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float value = -INFINITY;
    for (uint index = 0; index < 8; ++index) value = max(value, reduction[index]);
    maximum = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float localDenominator = 0.0f;
  for (uint sequencePosition = tid; sequencePosition < scoreCount;
       sequencePosition += 256) {
    const float value = fast::exp(scores[sequencePosition] - maximum);
    scores[sequencePosition] = value;
    localDenominator += value;
  }
  localDenominator = simd_sum(localDenominator);
  if (lane == 0) reduction[simdgroupIndex] = localDenominator;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float value = 0.0f;
    for (uint index = 0; index < 8; ++index) value += reduction[index];
    denominator = 1.0f / value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < latentDim) {
    float value = 0.0f;
    for (uint sequencePosition = 0; sequencePosition < scoreCount;
         ++sequencePosition) {
      const float latentValue = sequencePosition == position
          ? currentLatent[tid]
          : float(latentCache[
                (batch * maxSequence + sequencePosition) * latentDim + tid]);
      value += (scores[sequencePosition] * denominator) * latentValue;
    }
    headContext[tid] = bfloat(value);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const bfloat4 *headContext4 =
      reinterpret_cast<threadgroup bfloat4 *>(headContext);
  device const bfloat4 *outputWeight4 =
      reinterpret_cast<device const bfloat4 *>(outputWeight);
  const uint latentDim4 = latentDim / 4;
  for (uint out = tid; out < dim; out += 256) {
    float value = 0.0f;
    const uint weightOffset4 =
        (out * heads * latentDim + head * latentDim) / 4;
    for (uint d = 0; d < latentDim4; ++d) {
      value += dot(float4(headContext4[d]),
                   float4(outputWeight4[weightOffset4 + d]));
    }
    headOutput[(batch * heads + head) * dim + out] = value;
  }
}

kernel void attention_sum_heads_decode_bf16(
    device bfloat *output [[buffer(0)]],
    device const float *headOutput [[buffer(1)]],
    device const bfloat *addend [[buffer(2)]],
    constant uint &tokens [[buffer(3)]],
    constant uint &heads [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &addResidual [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= tokens * dim) return;
  const uint token = gid / dim;
  const uint out = gid - token * dim;
  float value = 0.0f;
  for (uint head = 0; head < heads; ++head) {
    value += headOutput[(token * heads + head) * dim + out];
  }
  const bfloat delta = bfloat(value);
  output[gid] = addResidual != 0
      ? bfloat(float(addend[gid]) + float(delta)) : delta;
}

kernel void vocab_argmax_partials_bf16(
    device float *partialValues [[buffer(0)]],
    device uint *partialIndices [[buffer(1)]],
    device const bfloat *hidden [[buffer(2)]],
    device const bfloat *weight [[buffer(3)]],
    constant uint &batchSize [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &vocabSize [[buffer(6)]],
    constant uint &groupCount [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint batch = group / groupCount;
  const uint tile = group - batch * groupCount;
  if (batch >= batchSize) return;
  const uint vocabIndex = tile * 256 + tid;
  float value = -INFINITY;
  if (vocabIndex < vocabSize) {
    device const bfloat4 *hidden4 =
        reinterpret_cast<device const bfloat4 *>(hidden);
    device const bfloat4 *weight4 =
        reinterpret_cast<device const bfloat4 *>(weight);
    const uint dim4 = dim / 4;
    value = 0.0f;
    for (uint d = 0; d < dim4; ++d) {
      value += dot(float4(hidden4[batch * dim4 + d]),
                   float4(weight4[vocabIndex * dim4 + d]));
    }
  }
  threadgroup float values[256];
  threadgroup uint indices[256];
  values[tid] = value;
  indices[tid] = vocabIndex;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other = values[tid + stride];
      const uint otherIndex = indices[tid + stride];
      if (other > values[tid] ||
          (other == values[tid] && otherIndex < indices[tid])) {
        values[tid] = other;
        indices[tid] = otherIndex;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    partialValues[batch * groupCount + tile] = values[0];
    partialIndices[batch * groupCount + tile] = indices[0];
  }
}

kernel void vocab_argmax_reduce_bf16(
    device long *output [[buffer(0)]],
    device const float *partialValues [[buffer(1)]],
    device const uint *partialIndices [[buffer(2)]],
    constant uint &batchSize [[buffer(3)]],
    constant uint &groupCount [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint batch [[threadgroup_position_in_grid]]) {
  if (batch >= batchSize) return;
  threadgroup float values[256];
  threadgroup uint indices[256];
  values[tid] = tid < groupCount
      ? partialValues[batch * groupCount + tid] : -INFINITY;
  indices[tid] = tid < groupCount
      ? partialIndices[batch * groupCount + tid] : 0xffffffffu;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other = values[tid + stride];
      const uint otherIndex = indices[tid + stride];
      if (other > values[tid] ||
          (other == values[tid] && otherIndex < indices[tid])) {
        values[tid] = other;
        indices[tid] = otherIndex;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) output[batch] = long(indices[0]);
}

kernel void mla_output_decode_bf16(
    device bfloat *output [[buffer(0)]],
    device const float *context [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const bfloat *addend [[buffer(3)]],
    constant uint &tokens [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &contextDim [[buffer(6)]],
    constant uint &addResidual [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= tokens * dim) return;
  const uint token = gid / dim;
  const uint out = gid - token * dim;
  float value = 0.0f;
  for (uint d = 0; d < contextDim; ++d) {
    value += context[token * contextDim + d] * weight[out * contextDim + d];
  }
  const bfloat delta = bfloat(value);
  output[gid] = addResidual != 0
      ? bfloat(float(addend[gid]) + float(delta)) : delta;
}
