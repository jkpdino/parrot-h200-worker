#include <metal_stdlib>
using namespace metal;
kernel void assignment_positions(device int *positions [[buffer(0)]],
    device atomic_uint *counts [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    constant uint &assignments [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= assignments) return;
    positions[i] = int(atomic_fetch_add_explicit(
        &counts[uint(indices[i])], 1u, memory_order_relaxed));
}
kernel void group_scatter(device float *grouped [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const int *positions [[buffer(3)]],
    constant uint &tokens [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant uint &topK [[buffer(6)]], constant uint &capacity [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    const uint assignment = i / dim;
    const uint d = i % dim;
    if (assignment >= tokens * topK) return;
    const uint slot = uint(positions[assignment]);
    if (slot < capacity) {
        const uint expert = uint(indices[assignment]);
        grouped[(expert * capacity + slot) * dim + d] =
            input[(assignment / topK) * dim + d];
    }
}
kernel void group_scatter_backward(device float *gradInput [[buffer(0)]],
    device const float *gradGrouped [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const int *positions [[buffer(3)]],
    constant uint &tokens [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant uint &topK [[buffer(6)]], constant uint &capacity [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= tokens * dim) return;
    const uint token = i / dim;
    const uint d = i % dim;
    float sum = 0.0f;
    for (uint k = 0; k < topK; ++k) {
        const uint assignment = token * topK + k;
        const uint slot = uint(positions[assignment]);
        if (slot < capacity) {
            const uint expert = uint(indices[assignment]);
            sum += float(gradGrouped[(expert * capacity + slot) * dim + d]);
        }
    }
    gradInput[i] = STORAGE(sum);
}
kernel void group_gather(device float *output [[buffer(0)]],
    device const float *expertOutput [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const int *positions [[buffer(3)]],
    device const float *weights [[buffer(4)]],
    constant uint &tokens [[buffer(5)]], constant uint &dim [[buffer(6)]],
    constant uint &topK [[buffer(7)]], constant uint &capacity [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= tokens * dim) return;
    const uint token = i / dim;
    const uint d = i % dim;
    float sum = 0.0f;
    for (uint k = 0; k < topK; ++k) {
        const uint assignment = token * topK + k;
        const uint slot = uint(positions[assignment]);
        if (slot < capacity) {
            const uint expert = uint(indices[assignment]);
            sum += float(expertOutput[(expert * capacity + slot) * dim + d]) *
                weights[assignment];
        }
    }
    output[i] = STORAGE(sum);
}
kernel void group_gather_backward_output(
    device float *gradExpert [[buffer(0)]],
    device const float *gradOutput [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device const int *positions [[buffer(3)]],
    device const float *weights [[buffer(4)]],
    constant uint &tokens [[buffer(5)]], constant uint &dim [[buffer(6)]],
    constant uint &topK [[buffer(7)]], constant uint &capacity [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    const uint assignment = i / dim;
    const uint d = i % dim;
    if (assignment >= tokens * topK) return;
    const uint slot = uint(positions[assignment]);
    if (slot < capacity) {
        const uint expert = uint(indices[assignment]);
        gradExpert[(expert * capacity + slot) * dim + d] = STORAGE(
            float(gradOutput[(assignment / topK) * dim + d]) * weights[assignment]);
    }
}
kernel void group_gather_backward_weight(
    device float *gradWeights [[buffer(0)]],
    device const float *gradOutput [[buffer(1)]],
    device const float *expertOutput [[buffer(2)]],
    device const long *indices [[buffer(3)]],
    device const int *positions [[buffer(4)]],
    constant uint &assignments [[buffer(5)]], constant uint &dim [[buffer(6)]],
    constant uint &topK [[buffer(7)]], constant uint &capacity [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint assignment [[threadgroup_position_in_grid]]) {
    if (assignment >= assignments) return;
    const uint slot = uint(positions[assignment]);
    float sum = 0.0f;
    if (slot < capacity) {
        const uint expert = uint(indices[assignment]);
        for (uint d = tid; d < dim; d += 256) {
            sum += float(gradOutput[(assignment / topK) * dim + d]) *
                float(expertOutput[(expert * capacity + slot) * dim + d]);
        }
    }
    sum = simd_sum(sum);
    threadgroup float partials[8];
    if (lane == 0) partials[simdgroupIndex] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint index = 0; index < 8; ++index) total += partials[index];
        gradWeights[assignment] = STORAGE(total);
    }
}

kernel void depth_training_scores(device SCORE *scores [[buffer(0)]],
    device SCORE *inverseRms [[buffer(1)]],
    device const float *values [[buffer(2)]],
    device const float *query [[buffer(3)]],
    device const float *normWeight [[buffer(4)]],
    constant uint &sources [[buffer(5)]], constant uint &tokens [[buffer(6)]],
    constant uint &dim [[buffer(7)]], constant float &eps [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint i [[threadgroup_position_in_grid]]) {
    if (i >= sources * tokens) return;
    const uint base = i * dim;
    float squareSum = 0.0f;
    float dotSum = 0.0f;
    for (uint d = tid; d < dim; d += 256) {
        const float x = float(values[base + d]);
        squareSum += x * x;
        dotSum += x * float(normWeight[d]) * float(query[d]);
    }
    squareSum = simd_sum(squareSum);
    dotSum = simd_sum(dotSum);
    threadgroup float squarePartials[8];
    threadgroup float dotPartials[8];
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
        const float inv = rsqrt(squareTotal / float(dim) + eps);
        scores[i] = dotTotal * inv;
        inverseRms[i] = inv;
    }
}
kernel void depth_training_softmax(device SCORE *weights [[buffer(0)]],
    device const SCORE *scores [[buffer(1)]],
    constant uint &sources [[buffer(2)]], constant uint &tokens [[buffer(3)]],
    uint token [[thread_position_in_grid]]) {
    if (token >= tokens) return;
    float maximum = scores[token];
    for (uint s = 1; s < sources; ++s)
        maximum = max(maximum, scores[s * tokens + token]);
    float denominator = 0.0f;
    for (uint s = 0; s < sources; ++s)
        denominator += exp(scores[s * tokens + token] - maximum);
    for (uint s = 0; s < sources; ++s)
        weights[s * tokens + token] =
            exp(scores[s * tokens + token] - maximum) / denominator;
}
kernel void depth_training_mix(device float *output [[buffer(0)]],
    device const SCORE *weights [[buffer(1)]],
    device const float *values [[buffer(2)]],
    constant uint &sources [[buffer(3)]], constant uint &tokens [[buffer(4)]],
    constant uint &dim [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= tokens * dim) return;
    const uint token = i / dim;
    const uint d = i % dim;
    float sum = 0.0f;
    for (uint s = 0; s < sources; ++s)
        sum += weights[s * tokens + token] *
            float(values[(s * tokens + token) * dim + d]);
    output[i] = STORAGE(sum);
}
kernel void depth_training_grad_scores(device SCORE *gradScores [[buffer(0)]],
    device const float *gradOutput [[buffer(1)]],
    device const float *values [[buffer(2)]], device const float *output [[buffer(3)]],
    device const SCORE *weights [[buffer(4)]],
    constant uint &sources [[buffer(5)]], constant uint &tokens [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint i [[threadgroup_position_in_grid]]) {
    if (i >= sources * tokens) return;
    const uint token = i % tokens;
    float dotValue = 0.0f;
    for (uint d = tid; d < dim; d += 256) {
        dotValue += float(gradOutput[token * dim + d]) *
            (float(values[i * dim + d]) - float(output[token * dim + d]));
    }
    dotValue = simd_sum(dotValue);
    threadgroup float partials[8];
    if (lane == 0) partials[simdgroupIndex] = dotValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint index = 0; index < 8; ++index) total += partials[index];
        gradScores[i] = weights[i] * total;
    }
}
kernel void depth_training_grad_values(device float *gradValues [[buffer(0)]],
    device const float *gradOutput [[buffer(1)]],
    device const float *values [[buffer(2)]], device const SCORE *scores [[buffer(3)]],
    device const SCORE *inverseRms [[buffer(4)]],
    device const SCORE *weights [[buffer(5)]],
    device const SCORE *gradScores [[buffer(6)]],
    device const float *query [[buffer(7)]],
    device const float *normWeight [[buffer(8)]],
    constant uint &sources [[buffer(9)]], constant uint &tokens [[buffer(10)]],
    constant uint &dim [[buffer(11)]], uint i [[thread_position_in_grid]]) {
    if (i >= sources * tokens * dim) return;
    const uint st = i / dim;
    const uint d = i % dim;
    const uint token = st % tokens;
    const float inv = inverseRms[st];
    const float x = float(values[i]);
    const float scoreGradient = gradScores[st];
    const float scoreInputGradient = inv * float(query[d]) * float(normWeight[d]) -
        inv * inv * x * scores[st] / float(dim);
    gradValues[i] = STORAGE(weights[st] * float(gradOutput[token * dim + d]) +
        scoreGradient * scoreInputGradient);
}
kernel void depth_training_grad_parameters(device float *gradQuery [[buffer(0)]],
    device float *gradNormWeight [[buffer(1)]],
    device const float *values [[buffer(2)]],
    device const SCORE *inverseRms [[buffer(3)]],
    device const SCORE *gradScores [[buffer(4)]],
    device const float *query [[buffer(5)]],
    device const float *normWeight [[buffer(6)]],
    constant uint &sources [[buffer(7)]], constant uint &tokens [[buffer(8)]],
    constant uint &dim [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint d [[threadgroup_position_in_grid]]) {
    if (d >= dim) return;
    float querySum = 0.0f;
    float normSum = 0.0f;
    for (uint st = tid; st < sources * tokens; st += 256) {
        const float common = gradScores[st] * float(values[st * dim + d]) *
            inverseRms[st];
        querySum += common * float(normWeight[d]);
        normSum += common * float(query[d]);
    }
    querySum = simd_sum(querySum);
    normSum = simd_sum(normSum);
    threadgroup float queryPartials[8];
    threadgroup float normPartials[8];
    if (lane == 0) {
        queryPartials[simdgroupIndex] = querySum;
        normPartials[simdgroupIndex] = normSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float queryTotal = 0.0f;
        float normTotal = 0.0f;
        for (uint index = 0; index < 8; ++index) {
            queryTotal += queryPartials[index];
            normTotal += normPartials[index];
        }
        gradQuery[d] = STORAGE(queryTotal);
        gradNormWeight[d] = STORAGE(normTotal);
    }
}
kernel void rms_training_forward(device float *output [[buffer(0)]],
    device SCORE *inverseRms [[buffer(1)]], device const float *input [[buffer(2)]],
    device const float *weight [[buffer(3)]], constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]], constant float &eps [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    if (row >= rows) return;
    const uint base = row * dim;
    float squareSum = 0.0f;
    for (uint d = tid; d < dim; d += 256) {
        const float x = float(input[base + d]);
        squareSum += x * x;
    }
    squareSum = simd_sum(squareSum);
    threadgroup float partials[8];
    threadgroup float inv;
    if (lane == 0) partials[simdgroupIndex] = squareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint index = 0; index < 8; ++index) total += partials[index];
        inv = rsqrt(total / float(dim) + eps);
        inverseRms[row] = inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < dim; d += 256)
        output[base + d] = STORAGE(
            float(input[base + d]) * inv * float(weight[d]));
}
kernel void rms_training_backward_input(device float *gradInput [[buffer(0)]],
    device const float *gradOutput [[buffer(1)]],
    device const float *input [[buffer(2)]], device const float *weight [[buffer(3)]],
    device const SCORE *inverseRms [[buffer(4)]],
    constant uint &rows [[buffer(5)]], constant uint &dim [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    if (row >= rows) return;
    const uint base = row * dim;
    float dot = 0.0f;
    for (uint d = tid; d < dim; d += 256)
        dot += float(gradOutput[base + d]) * float(input[base + d]) *
            float(weight[d]);
    dot = simd_sum(dot);
    threadgroup float partials[8];
    threadgroup float totalDot;
    if (lane == 0) partials[simdgroupIndex] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        totalDot = 0.0f;
        for (uint index = 0; index < 8; ++index) totalDot += partials[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inverseRms[row];
    for (uint d = tid; d < dim; d += 256) {
        const uint i = base + d;
        gradInput[i] = STORAGE(
            float(gradOutput[i]) * float(weight[d]) * inv -
            float(input[i]) * inv * inv * inv * totalDot / float(dim));
    }
}
kernel void rms_training_grad_weight(device float *gradWeight [[buffer(0)]],
    device const float *gradOutput [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device const SCORE *inverseRms [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &dim [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint d [[threadgroup_position_in_grid]]) {
    if (d >= dim) return;
    float sum = 0.0f;
    for (uint row = tid; row < rows; row += 256)
        sum += float(gradOutput[row * dim + d]) *
            float(input[row * dim + d]) * inverseRms[row];
    sum = simd_sum(sum);
    threadgroup float partials[8];
    if (lane == 0) partials[simdgroupIndex] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint index = 0; index < 8; ++index) total += partials[index];
        gradWeight[d] = STORAGE(total);
    }
}
kernel void cross_entropy_forward(device SCORE *rowLoss [[buffer(0)]],
    device SCORE *rowMaximum [[buffer(1)]],
    device SCORE *rowInverseSum [[buffer(2)]],
    device int *rowValid [[buffer(3)]],
    device const float *logits [[buffer(4)]],
    device const long *targets [[buffer(5)]],
    constant uint &batches [[buffer(6)]],
    constant uint &logitPositions [[buffer(7)]],
    constant uint &targetPositions [[buffer(8)]],
    constant uint &vocab [[buffer(9)]], constant uint &bagSize [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    if (row >= batches * targetPositions) return;
    const uint batch = row / targetPositions;
    const uint position = row % targetPositions;
    const uint inputBase = (batch * logitPositions + position) * vocab;
    float local = -INFINITY;
    for (uint token = lane; token < vocab; token += 256)
        local = max(local, float(logits[inputBase + token]));
    threadgroup float scratch[256];
    scratch[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] = max(scratch[lane], scratch[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float maximum = scratch[0];
    local = 0.0f;
    for (uint token = lane; token < vocab; token += 256)
        local += exp(float(logits[inputBase + token]) - maximum);
    scratch[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        const float sum = scratch[0];
        float targetSum = 0.0f;
        int valid = 0;
        for (uint k = 0; k < bagSize; ++k) {
            const long target = targets[row * bagSize + k];
            if (target != -100) {
                targetSum += float(logits[inputBase + uint(target)]);
                valid += 1;
            }
        }
        rowMaximum[row] = maximum;
        rowInverseSum[row] = 1.0f / sum;
        rowValid[row] = valid;
        rowLoss[row] = float(valid) * (log(sum) + maximum) - targetSum;
    }
}
kernel void cross_entropy_backward(device float *gradLogits [[buffer(0)]],
    device const float *logits [[buffer(1)]],
    device const long *targets [[buffer(2)]],
    device const SCORE *rowMaximum [[buffer(3)]],
    device const SCORE *rowInverseSum [[buffer(4)]],
    device const int *rowValid [[buffer(5)]],
    device const SCORE *denominator [[buffer(6)]],
    device const SCORE *gradLoss [[buffer(7)]],
    constant uint &batches [[buffer(8)]],
    constant uint &logitPositions [[buffer(9)]],
    constant uint &targetPositions [[buffer(10)]],
    constant uint &vocab [[buffer(11)]], constant uint &bagSize [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    const uint rows = batches * targetPositions;
    if (i >= rows * vocab) return;
    const uint row = i / vocab;
    const uint token = i % vocab;
    const uint batch = row / targetPositions;
    const uint position = row % targetPositions;
    const uint logitIndex = (batch * logitPositions + position) * vocab + token;
    int targetCount = 0;
    for (uint k = 0; k < bagSize; ++k)
        targetCount += int(targets[row * bagSize + k] == long(token));
    const float probability = exp(float(logits[logitIndex]) - rowMaximum[row]) *
        rowInverseSum[row];
    const float gradient = (probability * float(rowValid[row]) - float(targetCount)) /
        max(denominator[0], 1.0f) * gradLoss[0];
    gradLogits[logitIndex] = STORAGE(gradient);
}
kernel void swiglu_forward(device float *out [[buffer(0)]],
    device const float *gate [[buffer(1)]], device const float *up [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    float g = float(gate[i]);
    out[i] = STORAGE(g / (1.0f + exp(-g)) * float(up[i]));
}
kernel void swiglu_backward(device float *dg [[buffer(0)]],
    device float *du [[buffer(1)]], device const float *dy [[buffer(2)]],
    device const float *gate [[buffer(3)]], device const float *up [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    float g = float(gate[i]), u = float(up[i]), d = float(dy[i]);
    float s = 1.0f / (1.0f + exp(-g));
    dg[i] = STORAGE(d * u * s * (1.0f + g * (1.0f - s)));
    du[i] = STORAGE(d * g * s);
}
