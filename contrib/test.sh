#!/bin/sh

set -ex

FEATURES="compiler serde rand base64"

cargo --version
rustc --version

# Cache the toolchain we are using.
NIGHTLY=false
if cargo --version | grep nightly; then
    NIGHTLY=true
fi

# Format if told to
if [ "$DO_FMT" = true ]
then
    rustup component add rustfmt
    cargo fmt -- --check
fi

# Pin dependencies required to build with Rust 1.56.1
if cargo --version | grep "1\.56"; then
  cargo update -p secp256k1 --precise 0.28.1
  cargo update -p cc --precise 1.0.28
fi

# Test bitcoind integration tests if told to (this only works with the stable toolchain)
if [ "$DO_BITCOIND_TESTS" = true ]; then
    cd bitcoind-tests
    BITCOIND_EXE="$(git rev-parse --show-toplevel)/bitcoind-tests/bin/bitcoind" \
    cargo test --verbose

    # Exit integration tests, do not run other tests.
    exit 0
fi

# Defaults / sanity checks
cargo test

if [ "$DO_FEATURE_MATRIX" = true ]
then
    # All features
    cargo test --features="$FEATURES"

    # Single features
    for feature in ${FEATURES}
    do
        cargo test --features="$feature"
    done

# TODO: Pending examples
#    # Run all the examples
#    cargo build --examples
#    cargo run --example htlc --features=compiler
#    cargo run --example parse
#    cargo run --example sign_multisig
#    cargo run --example verify_tx > /dev/null
#    cargo run --example xpub_descriptors
#    cargo run --example taproot --features=compiler
#    cargo run --example psbt_sign_finalize --features=base64
fi

if [ "$DO_NO_STD" = true ]
then
  # Build no_std, to make sure that cfg(test) doesn't hide any issues
  cargo build --verbose --no-default-features --features="no-std"

  # Test no_std
  cargo test --verbose --no-default-features --features="no-std"

  # Build all features
  cargo build --verbose --no-default-features --features="no-std $FEATURES"

  # Build specific features
  for feature in ${FEATURES}
  do
      cargo build --verbose --no-default-features --features="no-std $feature"
  done
fi

# Bench if told to, only works with non-stable toolchain (nightly, beta).
if [ "$DO_BENCH" = true ]
then
    if [ "$NIGHTLY" = false ]; then
        if [ -n "$RUSTUP_TOOLCHAIN" ]; then
            echo "RUSTUP_TOOLCHAIN is set to a non-nightly toolchain but DO_BENCH requires a nightly toolchain"
        else
            echo "DO_BENCH requires a nightly toolchain"
        fi
        exit 1
    fi
    RUSTFLAGS='--cfg=bench' cargo bench
fi

# Build the docs if told to (this only works with the nightly toolchain)
if [ "$DO_DOCS" = true ]; then
    RUSTDOCFLAGS="--cfg docsrs" cargo +nightly rustdoc --features="$FEATURES" -- -D rustdoc::broken-intra-doc-links
fi

exit 0
