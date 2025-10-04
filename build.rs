fn main() {
    println!("cargo:rustc-link-search=native=src/engine/static");
    println!("cargo:rustc-link-lib=static=zerocopy");
}
