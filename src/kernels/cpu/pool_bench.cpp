#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const int nb = argc > 1 ? std::atoi(argv[1]) : 3, nt = argc > 2 ? std::atoi(argv[2]) : 1;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 2000, host_node = argc > 4 ? std::atoi(argv[4]) : 1;
    if (nb < 1 || nb > 32 || nt < 1 || nt > MAXT) { std::fprintf(stderr, "nb 1..32, nt 1..%d\n", MAXT); return 2; }
    NativeFmt f;
    std::string err;
    if (!native_fmt((int) GGML_TYPE_NVFP4, (int) GGML_TYPE_NVFP4, H, FF, f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    native_fmt_add_scales(f);
    std::vector<std::vector<int>> nodes = numa_physical_cores();
    if (nodes.size() < 2 || host_node >= (int) nodes.size() || nodes[(size_t) host_node].empty()) {
        std::fprintf(stderr, "needs two NUMA nodes\n");
        return 2;
    }
    const int hc = nodes[(size_t) host_node][0];
    nodes[(size_t) host_node].erase(nodes[(size_t) host_node].begin());
    if (const char* smt = std::getenv("STRATA_SMT"); smt != nullptr && smt[0] == '1') {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", hc);
        if (FILE* fl = std::fopen(path, "r")) {
            char buf[128] = {};
            if (std::fgets(buf, (int) sizeof buf, fl) != nullptr) {
                std::vector<int> sib;
                for (char* p = buf; *p != 0;) {
                    char* end = nullptr;
                    const long a = std::strtol(p, &end, 10);
                    if (end == p) { ++p; continue; }
                    long b = a;
                    p = end;
                    if (*p == '-') { b = std::strtol(p + 1, &end, 10); p = end; }
                    for (long c = a; c <= b; ++c) sib.push_back((int) c);
                }
                auto& hn = nodes[(size_t) host_node];
                hn.erase(std::remove_if(hn.begin(), hn.end(), [&](int c) { return std::find(sib.begin(), sib.end(), c) != sib.end(); }), hn.end());
            }
            std::fclose(fl);
        }
    }
    PoolNuma cfg;
    for (size_t n = 0; n < nodes.size(); ++n) {
        if ((int) n == host_node) cfg.host_group = (int) cfg.cores.size();
        cfg.cores.push_back(nodes[n]);
    }
    set_host_core(hc);
    bind_current_thread_to_node(host_node);
    pin_current_thread(hc);
    const int E = 48;
    const size_t bytes = (size_t) E * f.bytes;
    uint8_t* arena = (uint8_t*) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) { std::fprintf(stderr, "mmap failed\n"); return 1; }
    {
        const int G = (int) cfg.cores.size();
        std::vector<std::thread> th;
        for (int g = 0; g < G; ++g)
            th.emplace_back([&, g] {
                pin_current_thread(cfg.cores[(size_t) g][0]);
                std::mt19937 rng(1234 + g);
                const size_t f0 = (size_t) FF * g / G, f1 = (size_t) FF * (g + 1) / G, h0 = (size_t) H * g / G, h1 = (size_t) H * (g + 1) / G;
                for (int e = 0; e < E; ++e) {
                    uint8_t* b = arena + (size_t) e * f.bytes;
                    auto fill = [&](uint8_t* p, uint8_t* q, size_t row, size_t blocks) {
                        for (uint8_t* r = p; r < q; r += row)
                            for (size_t k = 0; k < blocks; ++k) {
                                uint8_t* blk = r + k * 36;
                                for (int s = 0; s < 4; ++s) blk[s] = (uint8_t) (0x30 + rng() % 16);
                                for (int j = 0; j < 32; ++j) blk[4 + j] = (uint8_t) rng();
                            }
                    };
                    fill(b + f0 * f.gu_row, b + f1 * f.gu_row, f.gu_row, H / 64);
                    fill(b + f.up_off + f0 * f.gu_row, b + f.up_off + f1 * f.gu_row, f.gu_row, H / 64);
                    fill(b + f.down_off + h0 * f.d_row, b + f.down_off + h1 * f.d_row, f.d_row, FF / 64);
                    if (g == 0) { const float s[4] = {0.01f, 0.01f, 0.01f, 0.f}; std::memcpy(b + f.scale_off, s, 16); }
                }
            });
        for (auto& t : th) t.join();
    }
    ExpertPool pool(0, true, true, &cfg);
    std::printf("pool: %d workers in %zu groups (+ the host on CPU %d, node %d)%s\n", pool.workers(), cfg.cores.size(), hc,
                host_node, std::getenv("STRATA_SMT") ? " [STRATA_SMT]" : "");
    std::mt19937 rng(28092026);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act((size_t) nt, std::vector<uint8_t>(kNativeActBytes));
    std::vector<float> x(H);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng);
        native_quant_act(f, x.data(), act[(size_t) t].data());
    }
    std::vector<float> out((size_t) nb * nt * H);
    std::vector<ExpertJobMulti> jobs((size_t) nb);
    int call = 0;
    auto set_jobs = [&] {
        for (int e = 0; e < nb; ++e) {
            ExpertJobMulti& j = jobs[(size_t) e];
            j.blob = arena + (size_t) ((call * 7 + e * 13) % E) * f.bytes;
            j.nt = nt;
            for (int t = 0; t < nt; ++t) { j.nact[t] = act[(size_t) t].data(); j.out[t] = out.data() + ((size_t) e * nt + t) * H; }
        }
        ++call;
    };
    for (int i = 0; i < 200; ++i) { set_jobs(); pool.run_split_multi_native(f, jobs.data(), nb); }
    pool.ms_multi_gu = pool.ms_multi_q = pool.ms_multi_down = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) { set_jobs(); pool.run_split_multi_native(f, jobs.data(), nb); }
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / iters;
    std::vector<float> ffb((size_t) nt * FF);
    std::vector<uint8_t> hq((size_t) nt * kNativeHBytes);
    const int reps = std::max(1, iters / 20);
    const auto t1 = Clock::now();
    for (int i = 0; i < reps; ++i) {
        set_jobs();
        for (int e = 0; e < nb; ++e) {
            float* ffp[MAXT];
            const void* hqp[MAXT];
            for (int t = 0; t < nt; ++t) { ffp[t] = ffb.data() + (size_t) t * FF; hqp[t] = hq.data() + (size_t) t * kNativeHBytes; }
            native_gu_rows(f, jobs[(size_t) e].blob, jobs[(size_t) e].nact, nt, ffp, 0, FF);
            for (int t = 0; t < nt; ++t) native_quant_h(f, ffp[t], hq.data() + (size_t) t * kNativeHBytes);
            native_down_rows(f, jobs[(size_t) e].blob, hqp, nt, jobs[(size_t) e].out, 0, H);
        }
    }
    const double one = std::chrono::duration<double, std::micro>(Clock::now() - t1).count() / reps;
    const int threads = pool.workers() + 1;
    const double mb = (double) nb * (double) f.bytes / 1e6;
    std::printf("%d experts x %d token(s) a layer (%.2f MB): pool %.1f us (%.0f GB/s; gate/up %.1f  quantize %.1f  down %.1f us)\n"
                "  one thread %.0f us -> / %d threads = %.1f us: the rest of the pool's time is synchronisation\n",
                nb, nt, mb, us, mb / us * 1e3, pool.ms_multi_gu * 1e3 / iters, pool.ms_multi_q * 1e3 / iters,
                pool.ms_multi_down * 1e3 / iters, one, threads, one / threads);
    munmap(arena, bytes);
    return 0;
}
