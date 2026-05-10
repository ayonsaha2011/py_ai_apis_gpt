FROM rust:1.94-bookworm AS build
WORKDIR /src
COPY gateway-rs ./gateway-rs
RUN cargo build --manifest-path gateway-rs/Cargo.toml --release

FROM node:22-bookworm AS frontend
WORKDIR /src/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/gateway-rs/target/release/gateway-rs /usr/local/bin/gateway-rs
COPY --from=frontend /src/frontend/dist ./frontend/dist
EXPOSE 8080
CMD ["gateway-rs"]
