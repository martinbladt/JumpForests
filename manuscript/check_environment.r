# Dependency/path checks only. Does not install packages or run any study.
if (!file.exists("SOURCE_MANIFEST.tsv")) {
    stop("Run this script with manuscript/ as the working directory.")
}
requirements <- read.delim("DEPENDENCIES.tsv", stringsAsFactors = FALSE)
available <- vapply(requirements$package, requireNamespace,
                    logical(1), quietly = TRUE)
if (any(!available)) {
    stop("Missing packages: ", paste(requirements$package[!available], collapse = ", "),
         ". See README.md for setup.")
}
versions <- vapply(requirements$package, function(package) {
    as.character(utils::packageVersion(package))
}, character(1))
print(data.frame(package = requirements$package,
                 reference = requirements$version,
                 installed = unname(versions)), row.names = FALSE)
if (any(versions != requirements$version)) {
    message("Some package versions differ from the recorded validation environment.")
}
if (is.null(get0("JFCppForestLeafVarianceMultistate",
                envir = asNamespace("JumpForests"), inherits = FALSE))) {
    stop("JumpForests lacks the native leaf-variance interface required for coverage.")
}
inputs <- c("data/biomedical_applications/cphholter.csv",
            "data/biomedical_applications/bmt.csv",
            "data/biomedical_applications/ebmt3.csv",
            "data/biomedical_applications/prova.csv")
if (any(!file.exists(inputs))) {
    stop("Bundled biomedical CSV inputs are missing:\n",
            paste(inputs[!file.exists(inputs)], collapse = "\n"))
}
message("Package, bundled-data and coverage-interface checks passed.")
