#include "benchkit/cuda_benchmarks.hpp"
#include "benchkit/statistics.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(                                                       \
                stderr,                                                         \
                "CUDA error at %s:%d: %s (%s)\n",                               \
                __FILE__,                                                       \
                __LINE__,                                                       \
                cudaGetErrorString(err_),                                       \
                cudaGetErrorName(err_));                                        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

namespace benchkit {

__global__ void tiny_kernel(float* x)
{
    int i = threadIdx.x;
    x[i] += 1.0f;
}

// Baseline for the CUDA Graphs experiment: measure the host-side cost of
// launching the same tiny kernel N times through the normal launch path
// (per-launch dispatch overhead + device execution), wall-clock timed.
BenchmarkResult run_cuda_graph_normal(
    const EnvironmentInfo& environment)
{
    constexpr int kDataElements = 32;
    constexpr int kLaunches = 1000;
    constexpr int kWarmup = 10;
    constexpr int kSamples = 20;

    const size_t bytes = kDataElements * sizeof(float);

    std::vector<float> h_x(kDataElements);

    for (int i = 0; i < kDataElements; ++i) {
        h_x[i] = static_cast<float>(i);
    }

    float* d_x = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMemcpy(
        d_x,
        h_x.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // Warmup
    for (int i = 0; i < kWarmup; ++i) {
        tiny_kernel<<<1, 32>>>(d_x);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Each sample times one batch of kLaunches sequential launches.
    std::vector<double> samples_ms;
    samples_ms.reserve(kSamples);

    for (int sample = 0; sample < kSamples; ++sample) {
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < kLaunches; ++i) {
            tiny_kernel<<<1, 32>>>(d_x);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::steady_clock::now();

        samples_ms.push_back(
            std::chrono::duration<double, std::milli>(
                end - start
            ).count()
        );
    }

    // Correctness: every launch adds 1.0f, so a correct run left
    // h_x[i] = i + kWarmup + kSamples * kLaunches.
    CUDA_CHECK(cudaMemcpy(
        h_x.data(),
        d_x,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool correct = true;

    const int expected_increments =
        kWarmup + kSamples * kLaunches;

    for (int i = 0; i < kDataElements; ++i) {
        if (h_x[i] !=
            static_cast<float>(i + expected_increments)) {
            correct = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_x));

    BenchmarkResult result;

    result.benchmark = "cuda_graph_normal";
    result.status = correct ? "pass" : "fail";
    result.environment = environment;
    result.sample_count = kSamples;
    result.raw_samples = samples_ms;

    result.measurements = calculate_statistics(
        samples_ms,
        "ms"
    );

    result.measurements.push_back({
        "launches_per_sample",
        "",
        static_cast<double>(kLaunches)
    });

    result.measurements.push_back({
        "kernel_executions_per_sample",
        "",
        static_cast<double>(kLaunches)
    });

    result.measurements.push_back({
        "total_kernel_executions",
        "",
        static_cast<double>(kWarmup + kSamples * kLaunches)
    });

    result.measurements.push_back({
        "elements",
        "",
        static_cast<double>(kDataElements)
    });

    return result;
}

// Graph-captured path: capture exactly ONE tiny_kernel launch into a CUDA
// Graph (nodes_per_graph = 1), instantiate it once, then replay the graph
// kReplaysPerSample times per sample — the same N as the normal path's
// launches per sample, so kernel executions per sample are identical for
// both paths. Setup (capture + instantiate) is host-timed separately from
// the replay loop so the one-time cost can be weighed against per-replay
// savings.
//
// Kept apples-to-apples with run_cuda_graph_normal: same kernel, same
// buffer, single stream, same warmup count (10), and the same host
// timing boundary ({launches/replays + device sync}) around each sample.
BenchmarkResult run_cuda_graph_captured(
    const EnvironmentInfo& environment)
{
    constexpr int kDataElements = 32;
    constexpr int kNodesPerGraph = 1;     // exactly one captured launch
    constexpr int kReplaysPerSample = 1000; // = N of the normal path
    constexpr int kWarmup = 10;           // warmup replays (same count as normal)
    constexpr int kSamples = 20;

    const size_t bytes = kDataElements * sizeof(float);

    std::vector<float> h_x(kDataElements);

    for (int i = 0; i < kDataElements; ++i) {
        h_x[i] = static_cast<float>(i);
    }

    float* d_x = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMemcpy(
        d_x,
        h_x.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // This driver rejects capture on the legacy NULL/default stream
    // (verified: both ThreadLocal and Global modes return
    // cudaErrorStreamCaptureUnsupported). The graph path therefore uses one
    // dedicated stream for capture AND replay — a single stream throughout,
    // matching the single-stream semantics of the normal baseline.
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));

    // ------------------------------------------------------------
    // Setup: capture the single launch, then instantiate.
    // Timed separately from replay so setup cost is visible.
    // ------------------------------------------------------------
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    auto setup_start = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaStreamBeginCapture(
        stream,
        cudaStreamCaptureModeThreadLocal
    ));

    tiny_kernel<<<1, 32, 0, stream>>>(d_x);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(
        &graph_exec,
        graph,
        nullptr,
        nullptr,
        0
    ));

    // The instantiated executable is self-contained; free the source graph.
    CUDA_CHECK(cudaGraphDestroy(graph));

    auto setup_end = std::chrono::steady_clock::now();

    const double setup_ms =
        std::chrono::duration<double, std::milli>(
            setup_end - setup_start
        ).count();

    // ------------------------------------------------------------
    // Warmup (same count as the normal path: 10 replays = 10 kernel
    // executions), then the same post-warmup error/sync checks.
    // ------------------------------------------------------------
    for (int i = 0; i < kWarmup; ++i) {
        cudaGraphLaunch(graph_exec, stream);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------
    // Timed replays: N (=1000) replays per sample, 20 samples.
    // Each replay executes the single captured kernel, so kernel
    // executions per sample equal the normal path's N.
    //
    // The timed loop is structurally identical to the normal path:
    // plain dispatches plus a post-loop device sync. The sync's
    // CUDA_CHECK surfaces any launch error exactly as the normal
    // path's (an async launch has no return code), so no per-replay
    // error plumbing runs inside the timed window.
    // ------------------------------------------------------------
    std::vector<double> replay_samples_ms;
    replay_samples_ms.reserve(kSamples);

    for (int sample = 0; sample < kSamples; ++sample) {
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < kReplaysPerSample; ++i) {
            cudaGraphLaunch(graph_exec, stream);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::steady_clock::now();

        replay_samples_ms.push_back(
            std::chrono::duration<double, std::milli>(
                end - start
            ).count()
        );
    }

    // ------------------------------------------------------------
    // Correctness: total kernel executions =
    // (warmup replays + timed replays) x nodes per graph.
    // kNodesPerGraph = 1 makes this identical to the normal path's total.
    // ------------------------------------------------------------
    const int expected_increments =
        (kWarmup + kSamples * kReplaysPerSample) * kNodesPerGraph;

    CUDA_CHECK(cudaMemcpy(
        h_x.data(),
        d_x,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    bool correct = true;

    for (int i = 0; i < kDataElements; ++i) {
        if (h_x[i] !=
            static_cast<float>(i + expected_increments)) {
            correct = false;
            break;
        }
    }

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_x));

    BenchmarkResult result;

    result.benchmark = "cuda_graph_captured";
    result.status = correct ? "pass" : "fail";
    result.environment = environment;
    result.sample_count = kSamples;
    result.raw_samples = replay_samples_ms;

    result.measurements = calculate_statistics(
        replay_samples_ms,
        "ms"
    );

    result.measurements.push_back({
        "setup_ms",
        "ms",
        setup_ms
    });

    result.measurements.push_back({
        "nodes_per_graph",
        "",
        static_cast<double>(kNodesPerGraph)
    });

    result.measurements.push_back({
        "replays_per_sample",
        "",
        static_cast<double>(kReplaysPerSample)
    });

    result.measurements.push_back({
        "kernel_executions_per_sample",
        "",
        static_cast<double>(kNodesPerGraph * kReplaysPerSample)
    });

    result.measurements.push_back({
        "replays_total",
        "",
        static_cast<double>(kSamples * kReplaysPerSample)
    });

    result.measurements.push_back({
        "total_kernel_executions",
        "",
        static_cast<double>(expected_increments)
    });

    return result;
}

} // namespace benchkit