// Copyright (C) 2026 Fulcrum Genomics LLC
// SPDX-License-Identifier: MIT

use prmi::verify_sa::certify_sa_order;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]
    #[test]
    fn prmi_gsa_matches_oracle(fwd in prop::collection::vec(0u8..=3, 1..200)) {
        prop_assert!(certify_sa_order(&fwd, 1).is_ok());
    }
}
