#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Mm.eg.db)
})

GO_SIMPLIFY_CUTOFF <- 0.7
GO_SIMPLIFY_MEASURE <- "Wang"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: run_clusterprofiler_enrichment.R <jobs.csv>", call. = FALSE)
}

jobs_path <- args[[1]]
jobs <- read.csv(jobs_path, stringsAsFactors = FALSE, check.names = FALSE)

# jobs.csv is a long-form gene-set table. Each row is one ranked gene for one
# enrichment job. Required columns:
# - job_id: stable identifier grouping rows into one enrichment query.
# - panel: analysis panel label, e.g. "ad_pig".
# - test: gene-set source, e.g. "DF" or "DF_norm".
# - variant: gene-set variant, e.g. "topN100".
# - rank: integer rank within the gene set; rows are sorted by this before use.
# - gene: mouse gene symbol to map through org.Mm.eg.db SYMBOL -> ENTREZID.
# - out_path: CSV path where this job's enrichment result should be written.
required <- c("job_id", "panel", "test", "variant", "rank", "gene", "out_path")
missing <- setdiff(required, colnames(jobs))
if (length(missing) > 0) {
  stop(sprintf("jobs file is missing columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
}

empty_result <- function() {
  data.frame(
    source = character(),
    native = character(),
    name = character(),
    display_name = character(),
    p_value = numeric(),
    p_raw = numeric(),
    p_adjust = numeric(),
    q_value = numeric(),
    significant = logical(),
    description = character(),
    term_size = numeric(),
    query_size = numeric(),
    intersection_size = numeric(),
    effective_domain_size = numeric(),
    precision = numeric(),
    recall = numeric(),
    query = character(),
    parents = character(),
    geneID = character(),
    hit_entrez = character(),
    hit_genes = character(),
    semantic_simplified = logical(),
    semantic_simplify_cutoff = numeric(),
    semantic_simplify_measure = character(),
    method = character(),
    stringsAsFactors = FALSE
  )
}

parse_ratio_num <- function(x) {
  parts <- strsplit(as.character(x), "/", fixed = TRUE)
  as.numeric(vapply(parts, function(p) if (length(p) >= 1) p[[1]] else NA_character_, character(1)))
}

parse_ratio_den <- function(x) {
  parts <- strsplit(as.character(x), "/", fixed = TRUE)
  as.numeric(vapply(parts, function(p) if (length(p) >= 2) p[[2]] else NA_character_, character(1)))
}

map_symbols_to_entrez <- function(symbols) {
  symbols <- unique(symbols[!is.na(symbols) & nzchar(symbols)])
  if (length(symbols) == 0) {
    return(data.frame(SYMBOL = character(), ENTREZID = character(), stringsAsFactors = FALSE))
  }
  mapped <- suppressMessages(
    AnnotationDbi::select(
      org.Mm.eg.db,
      keys = symbols,
      keytype = "SYMBOL",
      columns = c("SYMBOL", "ENTREZID")
    )
  )
  mapped <- mapped[!is.na(mapped$ENTREZID) & nzchar(mapped$ENTREZID), c("SYMBOL", "ENTREZID")]
  mapped <- mapped[!duplicated(mapped$ENTREZID), , drop = FALSE]
  mapped
}

standardize_term_name <- function(x) {
  x <- trimws(gsub("\\s+", " ", as.character(x)))
  vapply(x, function(term) {
    if (is.na(term) || !nzchar(term)) {
      return(term)
    }
    # GO BP descriptions are usually all lowercase. Uppercase only the first
    # character for display, but leave terms that begin with mixed-case
    # acronyms such as cGMP-PKG or mRNA untouched.
    if (!grepl("^[[:lower:]][[:upper:]]", term)) {
      term <- paste0(toupper(substr(term, 1, 1)), substr(term, 2, nchar(term)))
    }
    term
  }, character(1), USE.NAMES = FALSE)
}

standardize_result <- function(enrich_result, source_name, query_name, entrez_to_symbol, simplified_ids = NULL) {
  if (is.null(enrich_result)) {
    return(empty_result())
  }
  df <- as.data.frame(enrich_result)
  if (nrow(df) == 0) {
    return(empty_result())
  }
  native <- as.character(df$ID)
  if (source_name == "GO:BP" && !is.null(simplified_ids)) {
    semantic_simplified <- native %in% simplified_ids
    semantic_simplify_cutoff <- rep(GO_SIMPLIFY_CUTOFF, nrow(df))
    semantic_simplify_measure <- rep(GO_SIMPLIFY_MEASURE, nrow(df))
  } else if (source_name == "GO:BP") {
    semantic_simplified <- rep(NA, nrow(df))
    semantic_simplify_cutoff <- rep(GO_SIMPLIFY_CUTOFF, nrow(df))
    semantic_simplify_measure <- rep(GO_SIMPLIFY_MEASURE, nrow(df))
  } else {
    semantic_simplified <- rep(NA, nrow(df))
    semantic_simplify_cutoff <- rep(NA_real_, nrow(df))
    semantic_simplify_measure <- rep("", nrow(df))
  }
  term_size <- parse_ratio_num(df$BgRatio)
  effective_domain_size <- parse_ratio_den(df$BgRatio)
  query_size <- parse_ratio_den(df$GeneRatio)
  hit_genes <- vapply(strsplit(as.character(df$geneID), "/", fixed = TRUE), function(ids) {
    symbols <- unname(entrez_to_symbol[ids])
    symbols <- symbols[!is.na(symbols) & nzchar(symbols)]
    paste(unique(symbols), collapse = "/")
  }, character(1))
  out <- data.frame(
    source = source_name,
    native = native,
    name = as.character(df$Description),
    display_name = standardize_term_name(df$Description),
    p_value = as.numeric(df$p.adjust),
    p_raw = as.numeric(df$pvalue),
    p_adjust = as.numeric(df$p.adjust),
    q_value = as.numeric(df$qvalue),
    significant = !is.na(as.numeric(df$p.adjust)) &
      !is.na(as.numeric(df$qvalue)) &
      as.numeric(df$p.adjust) <= 0.05 &
      as.numeric(df$qvalue) <= 0.2,
    description = as.character(df$Description),
    term_size = term_size,
    query_size = query_size,
    intersection_size = as.numeric(df$Count),
    effective_domain_size = effective_domain_size,
    precision = as.numeric(df$Count) / query_size,
    recall = as.numeric(df$Count) / term_size,
    query = query_name,
    parents = "",
    geneID = as.character(df$geneID),
    hit_entrez = as.character(df$geneID),
    hit_genes = hit_genes,
    semantic_simplified = semantic_simplified,
    semantic_simplify_cutoff = semantic_simplify_cutoff,
    semantic_simplify_measure = semantic_simplify_measure,
    method = "clusterProfiler",
    stringsAsFactors = FALSE
  )
  out
}

run_one <- function(sub) {
  sub <- sub[order(sub$rank), , drop = FALSE]
  out_path <- sub$out_path[[1]]
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

  # Map gene symbols to Entrez IDs
  symbols <- unique(as.character(sub$gene))
  mapped <- map_symbols_to_entrez(symbols)
  if (nrow(mapped) < 5) {
    write.csv(empty_result(), out_path, row.names = FALSE, quote = TRUE)
    return(invisible(NULL))
  }
  entrez <- unique(as.character(mapped$ENTREZID))
  entrez_to_symbol <- setNames(mapped$SYMBOL, mapped$ENTREZID)
  query_name <- paste(sub$panel[[1]], sub$test[[1]], sub$variant[[1]], sep = "_")

  # Run enrichment analysis while keeping all overlapping terms
  go <- tryCatch(
    enrichGO(
      gene = entrez,
      OrgDb = org.Mm.eg.db,
      ont = "BP",
      pvalueCutoff = 1,
      qvalueCutoff = 1
    ),
    error = function(e) {
      message("enrichGO failed for ", query_name, ": ", conditionMessage(e))
      NULL
    }
  )
  go_simplified_ids <- NULL
  if (!is.null(go)) {
    # clusterProfiler::simplify uses GO semantic similarity to retain one
    # representative term from redundant GO neighborhoods. We keep all terms
    # in the output and only annotate which terms survived this reduction.
    go_simplified <- tryCatch(
      simplify(
        go,
        cutoff = GO_SIMPLIFY_CUTOFF,
        by = "p.adjust",
        select_fun = min,
        measure = GO_SIMPLIFY_MEASURE
      ),
      error = function(e) {
        message("simplify failed for ", query_name, ": ", conditionMessage(e))
        NULL
      }
    )
    if (!is.null(go_simplified)) {
      go_simplified_ids <- as.character(as.data.frame(go_simplified)$ID)
    }
  }
  kegg <- tryCatch(
    enrichKEGG(
      gene = entrez,
      organism = "mmu",
      pvalueCutoff = 1,
      qvalueCutoff = 1
    ),
    error = function(e) {
      message("enrichKEGG failed for ", query_name, ": ", conditionMessage(e))
      NULL
    }
  )

  # Combine the results
  result <- rbind(
    standardize_result(go, "GO:BP", query_name, entrez_to_symbol, go_simplified_ids),
    standardize_result(kegg, "KEGG", query_name, entrez_to_symbol)
  )
  if (nrow(result) > 0) {
    result <- result[order(result$p_value, result$p_raw, result$source), , drop = FALSE]
  }
  write.csv(result, out_path, row.names = FALSE, quote = TRUE)
  invisible(NULL)
}

for (job_id in unique(jobs$job_id)) {
  run_one(jobs[jobs$job_id == job_id, , drop = FALSE])
}
