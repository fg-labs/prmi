// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT
//
// C++ driver that exercises the full prmi C ABI.  Its primary job is to catch
// FFI shape regressions — it is NOT a tuned harness for real workloads — but the
// final `prmi_collect_smems` block additionally demonstrates the idiomatic C
// calling pattern for the fused per-read SMEM collector (opts setup, the
// grow-on-(-4) overflow retry, and iterating the returned SMEMs).
//
// Usage:
//   cpp_caller <sidecar_prefix> <32mer_hex_key> <query_2bit_file> \
//              <pac_unpacked_file> <pac_packed_file> <pac_num_bases>
//
// <32mer_hex_key>       : 16 hex digits (uint64_t, big-endian as-if).
// <query_2bit_file>     : binary file, 1 byte per base, values 0–3.
// <pac_unpacked_file>   : binary file, 1 byte per base, values 0–3.
// <pac_packed_file>     : binary file, 2 bits per base (BWA-MEME bntpac format).
// <pac_num_bases>       : decimal count of bases encoded in the packed file.

#include <prmi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

static std::vector<uint8_t> read_binary_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open file: %s\n", path);
        std::exit(1);
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<std::size_t>(sz));
    if (!f.read(reinterpret_cast<char*>(buf.data()), sz)) {
        std::fprintf(stderr, "read error: %s\n", path);
        std::exit(1);
    }
    return buf;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
            "usage: %s <sidecar_prefix> <32mer_hex_key> "
            "<query_2bit_file> <pac_unpacked_file> "
            "<pac_packed_file> <pac_num_bases>\n", argv[0]);
        return 1;
    }

    const char* prefix          = argv[1];
    const char* hex_key         = argv[2];
    const char* query_file      = argv[3];
    const char* pac_unpacked_f  = argv[4];
    const char* pac_packed_f    = argv[5];
    uint64_t    pac_num_bases   = std::strtoull(argv[6], nullptr, 10);

    // --- prmi_open -----------------------------------------------------------
    prmi_index_t* idx = nullptr;
    if (prmi_open(prefix, &idx) != 0) {
        std::fprintf(stderr, "prmi_open failed: %s\n",
                     prmi_last_error_message());
        return 1;
    }
    std::printf("opened: sa_num=%zu max_error_bound=%llu format=%s\n",
        prmi_sa_num(idx),
        static_cast<unsigned long long>(prmi_max_error_bound(idx)),
        prmi_format_version(idx));

    // --- prmi_lookup ---------------------------------------------------------
    uint64_t key = std::strtoull(hex_key, nullptr, 16);
    uint64_t pos = 0, err = 0;
    if (prmi_lookup(idx, key, &pos, &err) != 0) {
        std::fprintf(stderr, "prmi_lookup failed: %s\n",
                     prmi_last_error_message());
        prmi_close(idx);
        return 1;
    }
    std::printf("lookup: pos=%llu err=%llu\n",
        static_cast<unsigned long long>(pos),
        static_cast<unsigned long long>(err));

    // Read the query and pac inputs. `query` and the unpacked pac are consumed
    // by callers wiring up their own pipelines; the spectrum smoke below uses
    // the packed pac.
    std::vector<uint8_t> query        = read_binary_file(query_file);
    std::vector<uint8_t> pac_unpacked = read_binary_file(pac_unpacked_f);
    std::vector<uint8_t> pac_packed   = read_binary_file(pac_packed_f);
    (void)query;
    (void)pac_unpacked;

    // --- spectrum smoke -------------------------------------------------------
    // Build a 45-base read from packed reference positions 0..45.  The forward
    // spectrum query = read[5..45] (40 bases, starting at pivot=5), so there
    // are 5 bases of left context available for backward extension.
    // smoke.fa is ACGT×1024, so every base is guaranteed to match.
    {
        const int READ_LEN = 45;
        const int PIVOT    = 5;
        const int QLEN     = READ_LEN - PIVOT;  // 40

        // Reject a packed buffer too small for the declared base count before
        // indexing pac_packed[i / 4]: a mismatched <pac_num_bases> argument
        // would otherwise read out of bounds.
        const uint64_t packed_capacity_bases =
            static_cast<uint64_t>(pac_packed.size()) * 4ULL;
        if (pac_num_bases > packed_capacity_bases) {
            std::fprintf(stderr,
                "invalid input: pac_num_bases=%llu exceeds packed capacity=%llu bases\n",
                static_cast<unsigned long long>(pac_num_bases),
                static_cast<unsigned long long>(packed_capacity_bases));
            prmi_close(idx);
            return 1;
        }

        std::vector<uint8_t> read45(READ_LEN);
        for (int i = 0; i < READ_LEN && static_cast<uint64_t>(i) < pac_num_bases; ++i) {
            uint8_t byte  = pac_packed[i / 4];
            int     shift = 6 - 2 * (i % 4);
            read45[i] = (byte >> shift) & 0x3;
        }
        // query = read45[PIVOT..] = read45[5..45]
        const uint8_t* q_ptr = read45.data() + PIVOT;

        // ---- prmi_forward_spectrum -------------------------------------------
        uint64_t fwd_nsteps = 0;
        std::vector<prmi_smem_step_t> fwd_steps(static_cast<std::size_t>(QLEN));
        int rc_fwd = prmi_forward_spectrum(idx,
                                           q_ptr,
                                           QLEN,
                                           pac_packed.data(),
                                           pac_num_bases,
                                           fwd_steps.data(),
                                           static_cast<uint64_t>(QLEN),
                                           &fwd_nsteps);
        std::printf("forward_spectrum nsteps=%llu rc=%d",
            static_cast<unsigned long long>(fwd_nsteps), rc_fwd);
        if (fwd_nsteps > 0) {
            const prmi_smem_step_t& deep = fwd_steps[fwd_nsteps - 1];
            std::printf(" deepest_match_len=%llu deepest_occ=%llu",
                static_cast<unsigned long long>(deep.match_len),
                static_cast<unsigned long long>(deep.occ_count));
        }
        std::printf("\n");
        if (rc_fwd != 0) {
            std::fprintf(stderr, "prmi_forward_spectrum failed rc=%d: %s\n",
                         rc_fwd, prmi_last_error_message());
            prmi_close(idx);
            return 1;
        }
        if (fwd_nsteps < 1) {
            std::fprintf(stderr,
                "prmi_forward_spectrum: expected >=1 steps for reference query\n");
            prmi_close(idx);
            return 1;
        }

        // ---- prmi_backward_spectrum -----------------------------------------
        // Anchor = deepest forward step (max match_len).  read=read45, pivot=5;
        // backward extension can walk up to 5 bases left from the anchor.
        {
            const prmi_smem_step_t& deep = fwd_steps[fwd_nsteps - 1];
            uint64_t bwd_nsteps = 0;
            std::vector<prmi_smem_step_t> bwd_steps(static_cast<std::size_t>(PIVOT + 1));
            int rc_bwd = prmi_backward_spectrum(idx,
                                                deep.sa_start,
                                                deep.occ_count,
                                                deep.match_len,
                                                read45.data(),
                                                READ_LEN,
                                                PIVOT,
                                                pac_packed.data(),
                                                pac_num_bases,
                                                bwd_steps.data(),
                                                static_cast<uint64_t>(PIVOT + 1),
                                                &bwd_nsteps);
            std::printf("backward_spectrum nsteps=%llu rc=%d\n",
                static_cast<unsigned long long>(bwd_nsteps), rc_bwd);
            if (rc_bwd != 0) {
                std::fprintf(stderr, "prmi_backward_spectrum failed rc=%d: %s\n",
                             rc_bwd, prmi_last_error_message());
                prmi_close(idx);
                return 1;
            }
        }

        // ---- prmi_sa_positions ----------------------------------------------
        // Resolve the deepest forward step's SA interval to genome positions.
        // Print up to 8 positions.
        {
            const prmi_smem_step_t& deep = fwd_steps[fwd_nsteps - 1];
            if (deep.occ_count > 0) {
                uint64_t n_print = deep.occ_count < 8 ? deep.occ_count : 8;
                std::vector<uint64_t> positions(deep.occ_count);
                int rcs = prmi_sa_positions(idx, deep.sa_start, deep.occ_count,
                                            positions.data());
                if (rcs != 0) {
                    std::fprintf(stderr, "prmi_sa_positions failed: %s\n",
                                 prmi_last_error_message());
                    prmi_close(idx);
                    return 1;
                }
                std::printf("prmi_sa_positions: rc=0 count=%llu first_%llu_positions=",
                    static_cast<unsigned long long>(deep.occ_count),
                    static_cast<unsigned long long>(n_print));
                for (uint64_t i = 0; i < n_print; ++i) {
                    std::printf("%s%llu", (i > 0 ? "," : ""),
                                static_cast<unsigned long long>(positions[i]));
                }
                std::printf("\n");
            } else {
                // No match: call with count=0 to verify it returns 0 with NULL.
                int rcs = prmi_sa_positions(idx, 0, 0, nullptr);
                if (rcs != 0) {
                    std::fprintf(stderr,
                        "prmi_sa_positions(count=0) failed with rc=%d: %s\n",
                        rcs, prmi_last_error_message());
                    prmi_close(idx);
                    return 1;
                }
                std::printf("prmi_sa_positions: rc=0 count=0 (no match)\n");
            }
        }

        // ---- prmi_sa_positions_strided --------------------------------------
        // Fetch up to 4 positions from the deepest forward step, stride=1.
        {
            const prmi_smem_step_t& deep = fwd_steps[fwd_nsteps - 1];
            uint64_t n_fetch = deep.occ_count < 4 ? deep.occ_count : 4;
            std::vector<uint64_t> strided_pos(static_cast<std::size_t>(n_fetch));
            int rc_str = prmi_sa_positions_strided(idx,
                                                   deep.sa_start,
                                                   1,
                                                   n_fetch,
                                                   strided_pos.data());
            std::printf("sa_positions_strided rc=%d n_out=%llu",
                rc_str, static_cast<unsigned long long>(n_fetch));
            for (uint64_t i = 0; i < n_fetch; ++i) {
                std::printf(" pos[%llu]=%llu",
                    static_cast<unsigned long long>(i),
                    static_cast<unsigned long long>(strided_pos[i]));
            }
            std::printf("\n");
            if (rc_str != 0) {
                std::fprintf(stderr,
                    "prmi_sa_positions_strided failed rc=%d: %s\n",
                    rc_str, prmi_last_error_message());
                prmi_close(idx);
                return 1;
            }
        }

        // ---- prmi_forward_spectrum_batch ------------------------------------
        // 2-task batch over an arena: two copies of the same QLEN-base query.
        {
            const int NTASKS = 2;
            std::vector<uint8_t> q_arena(NTASKS * QLEN);
            for (int t = 0; t < NTASKS; ++t) {
                std::copy(q_ptr, q_ptr + QLEN, q_arena.begin() + t * QLEN);
            }
            prmi_fwd_task_t tasks[NTASKS];
            for (int t = 0; t < NTASKS; ++t) {
                tasks[t].query_off  = static_cast<uint64_t>(t * QLEN);
                tasks[t].query_len  = static_cast<uint32_t>(QLEN);
                tasks[t].steps_off  = static_cast<uint32_t>(t * QLEN);
                tasks[t].max_steps  = static_cast<uint32_t>(QLEN);
            }
            std::vector<prmi_smem_step_t> batch_steps(NTASKS * QLEN);
            std::vector<uint64_t>         batch_nsteps(NTASKS, 0);
            int rc_batch = prmi_forward_spectrum_batch(idx,
                                                       q_arena.data(),
                                                       static_cast<uint64_t>(q_arena.size()),
                                                       tasks,
                                                       static_cast<uint64_t>(NTASKS),
                                                       pac_packed.data(),
                                                       pac_num_bases,
                                                       batch_steps.data(),
                                                       static_cast<uint64_t>(batch_steps.size()),
                                                       batch_nsteps.data());
            std::printf("forward_spectrum_batch rc=%d", rc_batch);
            for (int t = 0; t < NTASKS; ++t) {
                std::printf(" [%d]nsteps=%llu", t,
                    static_cast<unsigned long long>(batch_nsteps[t]));
            }
            std::printf("\n");
            if (rc_batch != 0) {
                std::fprintf(stderr,
                    "prmi_forward_spectrum_batch failed rc=%d: %s\n",
                    rc_batch, prmi_last_error_message());
                prmi_close(idx);
                return 1;
            }
            // Both tasks must produce the same nsteps as the single call.
            for (int t = 0; t < NTASKS; ++t) {
                if (batch_nsteps[t] != fwd_nsteps) {
                    std::fprintf(stderr,
                        "forward_spectrum_batch task[%d] nsteps=%llu vs single=%llu\n",
                        t,
                        static_cast<unsigned long long>(batch_nsteps[t]),
                        static_cast<unsigned long long>(fwd_nsteps));
                    prmi_close(idx);
                    return 1;
                }
            }
            std::printf("forward_spectrum_batch: both tasks match single — OK\n");
        }
    }
    // --- end spectrum smoke ---------------------------------------------------

    // --- prmi_collect_smems usage --------------------------------------------
    // The fused per-read SMEM collector: one call returns all SMEMs for a read,
    // byte-identical to FMI seeding. This block shows the idiomatic C calling
    // pattern — opts setup, the grow-on-(-4) retry, and iterating the result.
    {
        const int READ_LEN = 60;  // 60 bases of ACGT×1024 reference => full match
        std::vector<uint8_t> read(READ_LEN);
        for (int i = 0; i < READ_LEN && static_cast<uint64_t>(i) < pac_num_bases; ++i) {
            uint8_t byte  = pac_packed[i / 4];
            int     shift = 6 - 2 * (i % 4);
            read[i] = (byte >> shift) & 0x3;
        }

        prmi_collect_opts_t opts;
        opts.min_seed_len  = 19;
        opts.split_len     = 28;
        opts.split_width   = 10;
        opts.max_mem_intv  = 20;  // > 0 => pass 3 (the long-MEM reseed round) enabled
        const uint32_t rid = 7;

        // Size-probe: out_cap=0 returns -4 with *out_n set to the required count
        // (a caller that prefers to size the buffer exactly before allocating).
        int needed = -1;
        int rc_probe = prmi_collect_smems(idx, read.data(), READ_LEN, rid, &opts,
                                          pac_packed.data(), pac_num_bases,
                                          nullptr, 0, &needed);
        std::printf("collect_smems size-probe rc=%d needed=%d\n", rc_probe, needed);
        if (rc_probe != 0 && rc_probe != -4) {
            std::fprintf(stderr, "prmi_collect_smems probe failed rc=%d: %s\n",
                         rc_probe, prmi_last_error_message());
            prmi_close(idx);
            return 1;
        }

        // Idiomatic call: start from a guess and grow to the reported count on -4.
        std::vector<prmi_smem_t> out(8);
        int n = 0;
        for (;;) {
            int rc = prmi_collect_smems(idx, read.data(), READ_LEN, rid, &opts,
                                        pac_packed.data(), pac_num_bases,
                                        out.data(), static_cast<int>(out.size()), &n);
            if (rc == -4) {                 // out too small: *n is the required count
                out.resize(static_cast<std::size_t>(n));
                continue;
            }
            if (rc != 0) {
                std::fprintf(stderr, "prmi_collect_smems failed rc=%d: %s\n",
                             rc, prmi_last_error_message());
                prmi_close(idx);
                return 1;
            }
            break;
        }
        std::printf("collect_smems n=%d", n);
        if (n > 0) {
            const prmi_smem_t& s = out[0];
            std::printf(" first=(rid=%u m=%u n=%u k=%lld s=%lld)",
                s.rid, s.m, s.n,
                static_cast<long long>(s.k), static_cast<long long>(s.s));
        }
        std::printf("\n");
        if (n < 1) {
            std::fprintf(stderr,
                "prmi_collect_smems: expected >=1 SMEM for a reference-derived read\n");
            prmi_close(idx);
            return 1;
        }
        // Every emitted SMEM must carry the requested rid and a valid span/interval.
        for (int i = 0; i < n; ++i) {
            const prmi_smem_t& s = out[static_cast<std::size_t>(i)];
            if (s.rid != rid || s.n < s.m || s.s < 1) {
                std::fprintf(stderr,
                    "prmi_collect_smems: malformed SMEM[%d] (rid=%u m=%u n=%u s=%lld)\n",
                    i, s.rid, s.m, s.n, static_cast<long long>(s.s));
                prmi_close(idx);
                return 1;
            }
        }
        std::printf("collect_smems: all %d SMEMs well-formed — OK\n", n);
    }
    // --- end prmi_collect_smems usage ----------------------------------------

    // --- prmi_close ----------------------------------------------------------
    prmi_close(idx);
    return 0;
}
