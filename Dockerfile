# TODO: any python code change will cause this to rebuild
# It should do the expensive `uv sync` and maturin build only when needed

# NOTE: MUST BE BUILT FROM ROOT OF REPO due to dependency on ../common

############################ BUILD IMAGE STAGE ############################
FROM ubuntu:24.04 AS builder

# Accept version from build args for setuptools-scm and TypeScript version script
ARG SETUPTOOLS_SCM_PRETEND_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for building the host module)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files (paths relative to parent directory context)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY test/ ./test/
COPY setup.py ./
COPY pre_build.py ./
COPY default_providers_config.yml ./default_providers_config.yml

# Compiling Python source files to bytecode is typically desirable for production images as it tends to improve startup time (at the cost of increased installation time).
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_INSTALL_DIR=/app/.python
# Set up PYTHONPATH to include src directories so imports work
ENV PYTHONPATH="/app/src:/app/common/src"

# Install all dependencies and build the main project (this will run setup.py build_py)
# a --no-editable flag, which instructs uv to install the project in non-editable mode, removing any dependency on the source code
# In the context of a multi-stage Docker image, --no-editable can be used to include the project in the synced virtual environment from one stage,
# then copy the virtual environment alone (and not the source code) into the final image.
RUN uv sync --locked --no-editable
RUN uv run --locked --no-editable pre_build.py
# Clean up build artifacts
RUN rm -rf src/sandbox/host/target
RUN rm -rf src/sandbox/build

# ############################ FINAL IMAGE STAGE ############################
FROM ubuntu:24.04

WORKDIR /app
 
COPY --from=builder /app/ /app

ENV PYTHONPATH="/app/src:/app/common/src"
ENV PYTHONUNBUFFERED=1

# Expose the default session manager port
EXPOSE 2345

# # Start the session manager server (auth configured via env vars)
# # Note: in prod this CMD is always overritten (see session_factory/src/flyio_app_service.py:228)
CMD [".venv/bin/python", "-m", "application.main", "--port=2345", "--log-level=DEBUG"]