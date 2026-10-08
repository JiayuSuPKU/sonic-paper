#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(clusterProfiler)
})

GO_SIMPLIFY_CUTOFF <- 0.7
GO_SIMPLIFY_MEASURE <- "Wang"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: run_clusterprofiler.R <jobs.csv>", call. = FALSE)
}

jobs <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
required <- c("job_id", "panel", "test", "variant", "rank", "gene", "out_path", "orgdb", "organism")
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

clean_scalar <- function(x) {
  value <- as.character(x[[1]])
  if (is.na(value)) "" else trimws(value)
}

parse_ratio_num <- function(x) {
  parts <- strsplit(as.character(x), "/", fixed = TRUE)
  as.numeric(vapply(parts, function(p) if (length(p) >= 1) p[[1]] else NA_character_, character(1)))
}

parse_ratio_den <- function(x) {
  parts <- strsplit(as.character(x), "/", fixed = TRUE)
  as.numeric(vapply(parts, function(p) if (length(p) >= 2) p[[2]] else NA_character_, character(1)))
}

load_orgdb <- function(orgdb_name) {
  suppressPackageStartupMessages(library(orgdb_name, character.only = TRUE))
  get(orgdb_name)
}

map_symbols_to_entrez <- function(symbols, orgdb) {
  symbols <- unique(symbols[!is.na(symbols) & nzchar(symbols)])
  if (length(symbols) == 0) {
    return(data.frame(SYMBOL = character(), ENTREZID = character(), stringsAsFactors = FALSE))
  }
  mapped <- suppressMessages(
    AnnotationDbi::select(
      orgdb,
      keys = symbols,
      keytype = "SYMBOL",
      columns = c("SYMBOL", "ENTREZID")
    )
  )
  mapped <- mapped[!is.na(mapped$ENTREZID) & nzchar(mapped$ENTREZID), c("SYMBOL", "ENTREZID")]
  mapped[!duplicated(mapped$ENTREZID), , drop = FALSE]
}

standardize_term_name <- function(x) {
  x <- trimws(gsub("\\s+", " ", as.character(x)))
  vapply(x, function(term) {
    if (is.na(term) || !nzchar(term)) {
      return(term)
    }
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
  query_size <- parse_ratio_den(df$GeneRatio)
  hit_genes <- vapply(strsplit(as.character(df$geneID), "/", fixed = TRUE), function(ids) {
    symbols <- unname(entrez_to_symbol[ids])
    symbols <- symbols[!is.na(symbols) & nzchar(symbols)]
    paste(unique(symbols), collapse = "/")
  }, character(1))
  p_raw <- as.numeric(df$pvalue)
  p_adjust <- as.numeric(df$p.adjust)
  q_value <- as.numeric(df$qvalue)
  data.frame(
    source = source_name,
    native = native,
    name = as.character(df$Description),
    display_name = standardize_term_name(df$Description),
    p_value = p_adjust,
    p_raw = p_raw,
    p_adjust = p_adjust,
    q_value = q_value,
    significant = !is.na(p_adjust) & !is.na(q_value) & p_adjust <= 0.05 & q_value <= 0.2,
    description = as.character(df$Description),
    term_size = term_size,
    query_size = query_size,
    intersection_size = as.numeric(df$Count),
    effective_domain_size = parse_ratio_den(df$BgRatio),
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
}

read_universe <- function(path) {
  if (!nzchar(path)) {
    return(NULL)
  }
  universe <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!("gene" %in% colnames(universe))) {
    stop(sprintf("universe file has no gene column: %s", path), call. = FALSE)
  }
  unique(as.character(universe$gene))
}

run_one <- function(sub) {
  sub <- sub[order(sub$rank), , drop = FALSE]
  out_path <- sub$out_path[[1]]
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

  orgdb_name <- clean_scalar(sub$orgdb)
  organism <- clean_scalar(sub$organism)
  universe_path <- if ("universe_path" %in% colnames(sub)) clean_scalar(sub$universe_path) else ""
  semantic_simplify <- !("semantic_simplify" %in% colnames(sub)) ||
    isTRUE(as.logical(sub$semantic_simplify[[1]]))
  semantic_simplify_p_adjust_cutoff <- if (
    "semantic_simplify_p_adjust_cutoff" %in% colnames(sub)
  ) as.numeric(sub$semantic_simplify_p_adjust_cutoff[[1]]) else 1
  orgdb <- load_orgdb(orgdb_name)
  mapped <- map_symbols_to_entrez(unique(as.character(sub$gene)), orgdb)
  universe_symbols <- read_universe(universe_path)
  universe_mapped <- if (is.null(universe_symbols)) NULL else map_symbols_to_entrez(universe_symbols, orgdb)
  if (nrow(mapped) < 5 || (!is.null(universe_mapped) && nrow(universe_mapped) < 10)) {
    write.csv(empty_result(), out_path, row.names = FALSE, quote = TRUE)
    return(invisible(NULL))
  }

  entrez <- unique(as.character(mapped$ENTREZID))
  universe_entrez <- if (is.null(universe_mapped)) NULL else unique(as.character(universe_mapped$ENTREZID))
  entrez_to_symbol <- setNames(mapped$SYMBOL, mapped$ENTREZID)
  query_name <- paste(sub$panel[[1]], sub$test[[1]], sub$variant[[1]], sep = "_")

  go_args <- list(
    gene = entrez,
    OrgDb = orgdb,
    keyType = "ENTREZID",
    ont = "BP",
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    pAdjustMethod = "BH"
  )
  if (!is.null(universe_entrez)) {
    go_args$universe <- universe_entrez
  }
  go <- tryCatch(
    do.call(enrichGO, go_args),
    error = function(e) {
      stop("enrichGO failed for ", query_name, ": ", conditionMessage(e), call. = FALSE)
    }
  )

  go_simplified_ids <- NULL
  if (!is.null(go) && semantic_simplify) {
    # Terms outside the display FDR cannot replace a significant term because
    # simplify() retains the minimum-p.adjust representative.
    go_for_simplify <- go
    go_for_simplify@result <- go_for_simplify@result[
      !is.na(go_for_simplify@result$p.adjust) &
        go_for_simplify@result$p.adjust <= semantic_simplify_p_adjust_cutoff,
      ,
      drop = FALSE
    ]
    if (nrow(go_for_simplify@result) > 0) {
      go_simplified <- simplify(
        go_for_simplify,
        cutoff = GO_SIMPLIFY_CUTOFF,
        by = "p.adjust",
        select_fun = min,
        measure = GO_SIMPLIFY_MEASURE
      )
      go_simplified_ids <- as.character(as.data.frame(go_simplified)$ID)
    } else {
      go_simplified_ids <- character()
    }
  }

  kegg <- NULL
  if (nzchar(organism)) {
    kegg_args <- list(
      gene = entrez,
      organism = organism,
      pvalueCutoff = 1,
      qvalueCutoff = 1,
      pAdjustMethod = "BH"
    )
    if (!is.null(universe_entrez)) {
      kegg_args$universe <- universe_entrez
    }
    kegg <- tryCatch(
      do.call(enrichKEGG, kegg_args),
      error = function(e) {
        stop("enrichKEGG failed for ", query_name, ": ", conditionMessage(e), call. = FALSE)
      }
    )
  }

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
