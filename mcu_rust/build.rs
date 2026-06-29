fn main() {
    println!("cargo:rerun-if-changed=src/cuda/matmul.cu");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    cc::Build::new()
        .cuda(true)
        .file("src/cuda/matmul.cu")
        .flag("-O3")
        .flag("-gencode")
        .flag("arch=compute_75,code=compute_75")
        .compile("mcu_cuda_matmul");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
