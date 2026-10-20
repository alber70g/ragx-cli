"""Default corpus globs: index source code and prose; skip lockfiles, dependency
manifests, licenses, minified and build output. gitignore-style patterns — a pattern
without a slash matches at any depth. Projects tailor these in ragx.toml."""

from __future__ import annotations

DOC_EXTS = ["md", "mdx", "markdown", "rst", "txt", "adoc", "org"]

SOURCE_EXTS = [
    "py", "pyi", "js", "jsx", "mjs", "cjs", "ts", "tsx", "mts", "cts", "vue", "svelte",
    "astro", "go", "rs", "java", "kt", "kts", "scala", "groovy", "c", "h", "cc", "cpp",
    "cxx", "hpp", "hh", "cs", "fs", "swift", "m", "mm", "rb", "php", "pl", "pm", "lua",
    "r", "R", "dart", "ex", "exs", "erl", "hrl", "clj", "cljs", "hs", "ml", "mli", "elm",
    "zig", "nim", "jl", "sh", "bash", "zsh", "fish", "ps1", "sql", "graphql", "gql",
    "proto", "tf", "hcl", "html", "css", "scss", "sass", "less",
]

DEFAULT_INCLUDE = [f"*.{ext}" for ext in DOC_EXTS + SOURCE_EXTS] + ["Dockerfile", "Makefile"]

DEFAULT_EXCLUDE = [
    # lockfiles and dependency manifests
    "*.lock", "bun.lockb", "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml",
    "go.sum", "requirements*.txt", "constraints*.txt",
    # licenses
    "LICENSE*", "COPYING*",
    # minified and generated output
    "*.min.js", "*.min.css", "*.map",
    "dist/", "build/", "target/", "coverage/", "vendor/",
]
