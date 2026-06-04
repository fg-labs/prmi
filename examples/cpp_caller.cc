// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT
//
// C++ driver that exercises the prmi C ABI surface — open, lookup, smem_range
// (packed/unpacked), sa_positions, batch, long-read, and minimizer.  Its only
// job is to catch FFI shape regressions — it is NOT a usage example for real
// workloads, and it does not call every helper (e.g. tokenize / reverse-
// complement are covered by the Rust-side FFI tests).
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

    // --- prmi_smem_range (unpacked) ------------------------------------------
    std::vector<uint8_t> query       = read_binary_file(query_file);
    std::vector<uint8_t> pac_unpacked = read_binary_file(pac_unpacked_f);

    // The C API requires exactly 32 bases per query (see prmi.h / README).
    // Validate once here: the query buffer is reused below for the batch and
    // long-read calls, where a short read would become an out-of-bounds
    // std::copy in this example rather than a clean FFI error.
    if (query.size() != 32) {
        std::fprintf(stderr, "query must contain exactly 32 bases, got %zu\n",
                     query.size());
        prmi_close(idx);
        return 1;
    }

    uint64_t k = 0, l = 0, s = 0;
    int rc = prmi_smem_range(idx,
                             query.data(),
                             static_cast<int>(query.size()),
                             pac_unpacked.data(),
                             pac_unpacked.size(),
                             &k, &l, &s);
    std::printf("smem_range(unpacked): rc=%d k=%llu l=%llu s=%llu\n",
        rc,
        static_cast<unsigned long long>(k),
        static_cast<unsigned long long>(l),
        static_cast<unsigned long long>(s));
    if (rc < 0) {
        std::fprintf(stderr, "prmi_smem_range failed: %s\n",
                     prmi_last_error_message());
        prmi_close(idx);
        return 1;
    }

    // --- prmi_smem_range_packed ----------------------------------------------
    std::vector<uint8_t> pac_packed = read_binary_file(pac_packed_f);

    uint64_t kp = 0, lp = 0, sp = 0;
    int rcp = prmi_smem_range_packed(idx,
                                     query.data(),
                                     static_cast<int>(query.size()),
                                     pac_packed.data(),
                                     pac_num_bases,
                                     &kp, &lp, &sp);
    std::printf("smem_range(packed):   rc=%d k=%llu l=%llu s=%llu\n",
        rcp,
        static_cast<unsigned long long>(kp),
        static_cast<unsigned long long>(lp),
        static_cast<unsigned long long>(sp));
    if (rcp < 0) {
        std::fprintf(stderr, "prmi_smem_range_packed failed: %s\n",
                     prmi_last_error_message());
        prmi_close(idx);
        return 1;
    }

    // --- Verify packed == unpacked -------------------------------------------
    if (k != kp || l != lp || s != sp) {
        std::fprintf(stderr,
            "MISMATCH: unpacked=(%llu,%llu,%llu) packed=(%llu,%llu,%llu)\n",
            static_cast<unsigned long long>(k),
            static_cast<unsigned long long>(l),
            static_cast<unsigned long long>(s),
            static_cast<unsigned long long>(kp),
            static_cast<unsigned long long>(lp),
            static_cast<unsigned long long>(sp));
        prmi_close(idx);
        return 1;
    }
    std::printf("smem_range: packed matches unpacked — OK\n");

    // --- prmi_sa_positions ---------------------------------------------------
    // Resolve the SA interval (kp, lp) from smem_range_packed to genome
    // positions. Print up to 8 positions.
    if (lp > 0) {
        uint64_t n_print = lp < 8 ? lp : 8;
        std::vector<uint64_t> positions(lp);
        int rcs = prmi_sa_positions(idx, kp, lp, positions.data());
        if (rcs != 0) {
            std::fprintf(stderr, "prmi_sa_positions failed: %s\n",
                         prmi_last_error_message());
            prmi_close(idx);
            return 1;
        }
        std::printf("prmi_sa_positions: rc=0 count=%llu first_%llu_positions=",
            static_cast<unsigned long long>(lp),
            static_cast<unsigned long long>(n_print));
        for (uint64_t i = 0; i < n_print; ++i) {
            std::printf("%s%llu", (i > 0 ? "," : ""),
                        static_cast<unsigned long long>(positions[i]));
        }
        std::printf("\n");
    } else {
        // No match: call with count=0 to verify it returns 0 with a NULL buffer.
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

    // --- prmi_smem_range_batch_packed ----------------------------------------
    // Build a batch of 4 queries: the same 32-base packed query repeated 4x.
    // Expected: all 4 slots produce the same (k, l, s) as the single-key call.
    {
        const int BATCH = 4;
        std::vector<uint8_t> batch_queries(BATCH * 32);
        for (int i = 0; i < BATCH; ++i) {
            std::copy(query.begin(), query.begin() + 32,
                      batch_queries.begin() + i * 32);
        }
        std::vector<uint64_t> bk(BATCH, 0), bl(BATCH, 0), bs(BATCH, 0);
        int rcb = prmi_smem_range_batch_packed(idx,
                                              batch_queries.data(),
                                              static_cast<uint64_t>(BATCH),
                                              pac_packed.data(),
                                              pac_num_bases,
                                              bk.data(), bl.data(), bs.data());
        std::printf("prmi_smem_range_batch_packed: rc=%d", rcb);
        for (int i = 0; i < BATCH; ++i) {
            std::printf(" [%d]k=%llu,l=%llu,s=%llu", i,
                static_cast<unsigned long long>(bk[i]),
                static_cast<unsigned long long>(bl[i]),
                static_cast<unsigned long long>(bs[i]));
        }
        std::printf("\n");
        if (rcb < 0) {
            std::fprintf(stderr, "prmi_smem_range_batch_packed failed: %s\n",
                         prmi_last_error_message());
            prmi_close(idx);
            return 1;
        }
        // All 4 slots must be identical and match the single-key packed result.
        for (int i = 0; i < BATCH; ++i) {
            if (bk[i] != kp || bl[i] != lp || bs[i] != sp) {
                std::fprintf(stderr,
                    "MISMATCH batch[%d]: batch=(%llu,%llu,%llu) "
                    "single=(%llu,%llu,%llu)\n",
                    i,
                    static_cast<unsigned long long>(bk[i]),
                    static_cast<unsigned long long>(bl[i]),
                    static_cast<unsigned long long>(bs[i]),
                    static_cast<unsigned long long>(kp),
                    static_cast<unsigned long long>(lp),
                    static_cast<unsigned long long>(sp));
                prmi_close(idx);
                return 1;
            }
        }
        std::printf("prmi_smem_range_batch_packed: all %d slots match single-key — OK\n",
                    BATCH);
    }

    // --- prmi_smem_range_long_read_packed ------------------------------------
    // Demonstrate long-read seeding on a synthetic 200-base read built from the
    // first 200 bases of the packed reference. Seed at 5 pivot offsets.
    {
        const uint64_t READ_LEN   = 200;
        const int      NPIVOTS    = 5;

        // Unpack READ_LEN bases from pac_packed into a 1-base-per-byte buffer.
        std::vector<uint8_t> lr_read(READ_LEN);
        for (uint64_t i = 0; i < READ_LEN && i < pac_num_bases; ++i) {
            uint8_t byte  = pac_packed[i / 4];
            int     shift = 6 - 2 * static_cast<int>(i % 4);
            lr_read[i] = (byte >> shift) & 0x3;
        }

        // Five pivot offsets spread evenly: 0, 40, 80, 120, 160.
        uint64_t pivots[NPIVOTS] = {0, 40, 80, 120, 160};
        uint64_t lr_k[NPIVOTS] = {}, lr_l[NPIVOTS] = {}, lr_s[NPIVOTS] = {};

        int rc_lr = prmi_smem_range_long_read_packed(
            idx,
            lr_read.data(),
            READ_LEN,
            pivots,
            static_cast<uint64_t>(NPIVOTS),
            pac_packed.data(),
            pac_num_bases,
            lr_k, lr_l, lr_s);

        std::printf("prmi_smem_range_long_read_packed: rc=%d", rc_lr);
        for (int i = 0; i < NPIVOTS; ++i) {
            std::printf(" [%d]pivot=%llu,k=%llu,l=%llu,s=%llu",
                i,
                static_cast<unsigned long long>(pivots[i]),
                static_cast<unsigned long long>(lr_k[i]),
                static_cast<unsigned long long>(lr_l[i]),
                static_cast<unsigned long long>(lr_s[i]));
        }
        std::printf("\n");
        if (rc_lr < 0) {
            std::fprintf(stderr, "prmi_smem_range_long_read_packed failed: %s\n",
                         prmi_last_error_message());
            prmi_close(idx);
            return 1;
        }
    }

    // --- prmi_minimizer_32mer ------------------------------------------------
    // Extract the lex-min 32-mer from the first 100 bases of the packed reference.
    {
        const uint64_t WIN = 100;
        std::vector<uint8_t> win_bases(WIN);
        for (uint64_t i = 0; i < WIN && i < pac_num_bases; ++i) {
            uint8_t byte  = pac_packed[i / 4];
            int     shift = 6 - 2 * static_cast<int>(i % 4);
            win_bases[i] = (byte >> shift) & 0x3;
        }
        uint64_t min_key = 0, min_off = 0;
        int rc_min = prmi_minimizer_32mer(win_bases.data(), WIN, &min_key, &min_off);
        std::printf("prmi_minimizer_32mer: rc=%d key=0x%016llx offset=%llu\n",
            rc_min,
            static_cast<unsigned long long>(min_key),
            static_cast<unsigned long long>(min_off));
    }

    // --- prmi_close ----------------------------------------------------------
    prmi_close(idx);
    return 0;
}
