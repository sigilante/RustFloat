use std::path::{Path, PathBuf};

fn c_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "c"))
        .collect();
    v.sort();
    v
}

fn main() {
    // SoftFloat comes from the sibling oracle crate's vendor tree so the two
    // copies cannot drift.
    let sf = Path::new("../sdfloat-oracle/vendor/SoftFloat/source");
    let sf_platform = Path::new("../sdfloat-oracle/csrc");
    let sb = Path::new("vendor/SoftBLAS");
    let mut build = cc::Build::new();
    build
        .include(sf_platform)
        .include(sf.join("include"))
        .include(sf.join("ARM-VFPv2"))
        .include(sb.join("include"))
        // softblas.h includes softfloat.h without platform.h, so the
        // thread-local storage class must reach every translation unit.
        .define("THREAD_LOCAL", "_Thread_local")
        .define("SOFTFLOAT_FAST_INT64", None)
        .define("SOFTFLOAT_ROUND_ODD", None)
        .define("INLINE_LEVEL", "5")
        .define("SOFTFLOAT_FAST_DIV32TO16", None)
        .define("SOFTFLOAT_FAST_DIV64TO32", None)
        .warnings(false)
        .opt_level(2); // SoftFloat's header-inline primitives need inlining
    for dir in [sf.to_path_buf(), sf.join("ARM-VFPv2")] {
        for f in c_files(&dir) {
            build.file(f);
        }
    }
    for level in ["level1", "level2", "level3"] {
        for f in c_files(&sb.join("src/blas").join(level)) {
            build.file(f);
        }
    }
    build.compile("sdblas_oracle");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=../sdfloat-oracle/vendor");
    println!("cargo:rerun-if-changed=../sdfloat-oracle/csrc");
}
