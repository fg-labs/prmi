// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT
//
// C++ driver that exercises the full prmi C ABI.  Its only job is to catch FFI
// shape regressions — it is NOT a usage example for real workloads.
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

    // --- prmi_close ----------------------------------------------------------
    prmi_close(idx);
    return 0;
}
