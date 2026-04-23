
library(limma)

cat("Loading LUAD and LUSC expression data...\n")

luad <- read.table("datasets/LUAD_HiSeqV2",
                   header = TRUE, sep = "\t",
                   row.names = 1, check.names = FALSE)

lusc <- read.table("datasets/LUSC_HiSeqV2",
                   header = TRUE, sep = "\t",
                   row.names = 1, check.names = FALSE)

cat(sprintf("LUAD: %d genes x %d samples\n", nrow(luad), ncol(luad)))
cat(sprintf("LUSC: %d genes x %d samples\n", nrow(lusc), ncol(lusc)))

common_genes <- intersect(rownames(luad), rownames(lusc))
cat(sprintf("Common genes: %d\n", length(common_genes)))

expr <- cbind(luad[common_genes, ], lusc[common_genes, ])
group <- factor(c(rep("LUAD", ncol(luad)), rep("LUSC", ncol(lusc))))

cat(sprintf("Combined matrix: %d genes x %d samples\n", nrow(expr), ncol(expr)))
keep <- apply(expr, 1, function(x) {
  median(x[group == "LUAD"]) > 1 | median(x[group == "LUSC"]) > 1
})
expr_filtered <- expr[keep, ]
cat(sprintf("Genes after low-expression filter: %d\n", nrow(expr_filtered)))

cat("Running limma...\n")

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

fit <- lmFit(as.matrix(expr_filtered), design)

contrast_matrix <- makeContrasts(LUSC - LUAD, levels = design)
fit2 <- contrasts.fit(fit, contrast_matrix)
fit2 <- eBayes(fit2)

limma_results <- topTable(fit2, number = Inf, sort.by = "B")
limma_results$gene <- rownames(limma_results)

cat(sprintf("limma complete. Top gene: %s (adj.P=%.2e)\n",
            limma_results$gene[1], limma_results$adj.P.Val[1]))
cat("Loading plaintext DE scores...\n")

plain <- read.csv("scoring/dataset2/d2_de_baseline_batch_c.csv",
                  row.names = 1)
cat("Columns in plaintext baseline:", colnames(plain), "\n")

plain_scores <- plain[, 1, drop = FALSE]
colnames(plain_scores) <- "mean_diff"
plain_scores$gene <- rownames(plain_scores)

#matching sequence
common <- intersect(plain_scores$gene, limma_results$gene)
cat(sprintf("Genes matchable between plaintext scores and limma: %d\n",
            length(common)))

if (length(common) < 10) {
  cat("\nVery few genes matched by name.\n")
}

if (length(common) >= 200) {

  plain_matched  <- plain_scores[plain_scores$gene %in% common, ]
  plain_matched  <- plain_matched[order(plain_matched$mean_diff,
                                        decreasing = TRUE), ]
  top200_plain   <- head(plain_matched$gene, 200)

  limma_matched  <- limma_results[limma_results$gene %in% common, ]
  limma_matched  <- limma_matched[order(abs(limma_matched$logFC),
                                        decreasing = TRUE), ]
  top200_limma   <- head(limma_matched$gene, 200)

  overlap        <- intersect(top200_plain, top200_limma)
  overlap_pct    <- round(length(overlap) / 200 * 100, 1)

  cat("\n============================================================\n")
  cat(sprintf("TOP-200 OVERLAP: %d / 200 genes (%.1f%%)\n",
              length(overlap), overlap_pct))
  cat("============================================================\n\n")

  plain_vec <- plain_matched$mean_diff[match(common, plain_matched$gene)]
  limma_vec <- abs(limma_matched$logFC[match(common, limma_matched$gene)])
  rho       <- cor(plain_vec, limma_vec, method = "spearman",
                   use = "complete.obs")
  cat(sprintf("\nSpearman rho (mean-diff vs |limma logFC|): %.4f\n", rho))
} else {
  cat("\nFalling back to rank correlation on matched genes...\n")

  if (length(common) > 0) {
    plain_vec <- plain_scores$mean_diff[match(common, plain_scores$gene)]
    limma_vec <- abs(limma_results$logFC[match(common, limma_results$gene)])
    rho       <- cor(plain_vec, limma_vec, method = "spearman",
                     use = "complete.obs")
    cat(sprintf("Spearman rho (mean-diff vs |limma logFC|): %.4f\n", rho))
    cat(sprintf("Based on %d matched genes.\n", length(common)))
  } else {
    cat("No genes matched.\n")
  }
}

dir.create("results", showWarnings = FALSE)
limma_out <- limma_results[, c("gene", "logFC", "AveExpr",
                                "t", "P.Value", "adj.P.Val", "B")]
write.csv(limma_out,
          "results/limma_validation_results.csv",
          row.names = FALSE)
cat("\nSaved: results/limma_validation_results.csv\n")
cat("Done.\n")
d2 <- read.csv("scoring/dataset2/d2_de_baseline_batch_c.csv", row.names=1)
cat("D2 baseline columns:", colnames(d2), "\n")
cat("D2 sample gene IDs:", head(rownames(d2), 5), "\n")
cat("D2 genes:", nrow(d2), "\n")
common <- intersect(rownames(d2), limma_results$gene)
cat("Genes matching limma:", length(common), "\n")
