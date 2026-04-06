#!/bin/bash
set -ex

# Ensure Rust is in PATH
source $HOME/.cargo/env

# Print versions

java -version
rustc --version
cargo --version
cargo ndk --version
$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager --version

echo "Devcontainer setup complete."
