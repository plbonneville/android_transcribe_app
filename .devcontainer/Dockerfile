# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk wget unzip git curl build-essential pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Android SDK/NDK
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV ANDROID_NDK_VERSION=28.0.13004108
RUN mkdir -p /opt/android-sdk && \
    cd /opt/android-sdk && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O cmdline-tools.zip && \
    unzip cmdline-tools.zip && \
    rm cmdline-tools.zip && \
    mv cmdline-tools cmdline-tools-temp && \
    mkdir -p cmdline-tools/latest && \
    mv cmdline-tools-temp/* cmdline-tools/latest/ && \
    rmdir cmdline-tools-temp && \
    yes | /opt/android-sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root=/opt/android-sdk "platform-tools" "platforms;android-34" "build-tools;34.0.0" "ndk;$ANDROID_NDK_VERSION"


# Fix permissions so vscode user can write to SDK/NDK
RUN chown -R vscode:vscode /opt/android-sdk

# Add Android tools to PATH
ENV PATH="$PATH:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/build-tools/34.0.0:$ANDROID_SDK_ROOT/ndk/$ANDROID_NDK_VERSION"

# Install Rust as vscode user
USER vscode
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . $HOME/.cargo/env && \
    rustup target add aarch64-linux-android && \
    cargo install cargo-ndk
ENV PATH="$PATH:/home/vscode/.cargo/bin:$PATH"
USER root

# Set JAVA_HOME for Gradle
# Ensure both paths are valid by symlinking /opt/android-studio/jbr to /usr/lib/jvm/java-17-openjdk-amd64
RUN mkdir -p /opt/android-studio && ln -sf /usr/lib/jvm/java-17-openjdk-amd64 /opt/android-studio/jbr
ENV JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"

# Default workdir
WORKDIR /workspaces/android_transcribe_app