use std::path::Path;

fn main() {
    let src = Path::new("vendor/SoftFloat/source");
    let mut build = cc::Build::new();
    build
        .include("csrc")
        .include(src.join("include"))
        .include(src.join("ARM-VFPv2"))
        .define("SOFTFLOAT_FAST_INT64", None)
        .define("SOFTFLOAT_ROUND_ODD", None)
        .define("INLINE_LEVEL", "5")
        .define("SOFTFLOAT_FAST_DIV32TO16", None)
        .define("SOFTFLOAT_FAST_DIV64TO32", None)
        .warnings(false)
        // SoftFloat's `INLINE` primitives are C99 `inline` definitions in
        // headers with no out-of-line copy; they only link when inlined,
        // so the C must be optimized even in a debug Rust build.
        .opt_level(2)
        .file("csrc/shim.c");
    // csrc/hoon-overflow replaces the vendored files of the same name.
    let hoon = Path::new("csrc/hoon-overflow");
    for dir in [src.to_path_buf(), src.join("ARM-VFPv2")] {
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "c"))
            .collect();
        files.sort();
        for f in files {
            let patched = hoon.join(f.file_name().unwrap());
            build.file(if patched.exists() { patched } else { f });
        }
    }
    build.compile("sdfloat_oracle");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=vendor");
}
