# Prepare the 440-person data used for the Beaumont et al. (2002) comparison.
# Run from this directory: Rscript prepare_data.R
# Source archive: https://web.stanford.edu/group/pritchardlab/dataArchive/ydata.html

source_file <- "data/ydata-source.html"
stopifnot(file.exists(source_file))
html <- paste(readLines(source_file, warn = FALSE, encoding = "latin1"),
              collapse = "\n")
pattern <- '(?s)<TT><FONT COLOR="#000000"><FONT SIZE=\\+1>.*?</TT>'
chunks <- regmatches(html, gregexpr(pattern, html, perl = TRUE))[[1]]
plain <- trimws(gsub("<[^>]*>", " ",
                      gsub("&nbsp;", " ", chunks, fixed = TRUE)))
fields <- strsplit(plain, "[[:space:]]+")
is_record <- vapply(fields, function(f)
  length(f) == 9L && all(grepl("^[0-9]+$", f[-1L])), logical(1))
records <- fields[is_record]
stopifnot(length(records) == 445L)

ids <- vapply(records, `[[`, "", 1L)
stopifnot(!anyDuplicated(ids))
fragments <- matrix(as.integer(unlist(lapply(records, `[`, -1L))),
                    ncol = 8L, byrow = TRUE)
colnames(fragments) <- c("DYS19", "DYS388", "DYS389a", "DYS389c",
                         "DYS390", "DYS391", "DYS392", "DYS393")

# The archive labels the second DYS389 component "389c": it has already
# been separated from 389a. No second subtraction is applied.
repeat_unit <- c(4L, 3L, 4L, 4L, 4L, 4L, 3L, 4L)
offset <- c(0L, 2L, 1L, 0L, 0L, 0L, 1L, 0L)
remainder <- sweep(fragments, 2L, repeat_unit, `%%`)
fractional <- rowSums(sweep(remainder, 2L, offset, `!=`)) > 0L
stopifnot(sum(fractional) == 5L,
          all(remainder[!fractional, ] == matrix(offset,
            nrow = sum(!fractional), ncol = 8L, byrow = TRUE)))

excluded <- data.frame(source_id = ids[fractional],
                       reason = "DYS389c fragment length is 99, three bases off the four-base repeat lattice")
write.csv(excluded, "data/excluded_rows.csv", row.names = FALSE)

# Marker-specific offsets are arbitrary ancestral repeat origins. Each
# column is integer-valued after conversion; variances and profile counts
# are invariant to these offsets.
repeat_counts <- sweep(sweep(fragments[!fractional, , drop = FALSE],
                             2L, offset, `-`), 2L, repeat_unit, `/`)
stopifnot(all(repeat_counts == round(repeat_counts)))
storage.mode(repeat_counts) <- "integer"
write.csv(repeat_counts, "data/observed_repeat_counts.csv", row.names = FALSE)
write.csv(data.frame(source_id = ids[!fractional]),
          "data/retained_source_ids.csv", row.names = FALSE)

variance <- mean(apply(repeat_counts, 2L, var))
diversity <- mean(vapply(seq_len(ncol(repeat_counts)), function(j) {
  p <- prop.table(table(repeat_counts[, j]))
  1 - sum(p^2)
}, 0.0))
profiles <- nrow(unique(as.data.frame(repeat_counts)))
stopifnot(nrow(repeat_counts) == 440L,
          abs(variance - 1.123) < 0.0005,
          abs(diversity - 0.635) < 0.0005,
          profiles == 312L)
cat(sprintf("Retained %d of %d; variance %.6f; diversity %.6f; profiles %d\n",
            nrow(repeat_counts), length(records), variance, diversity, profiles))
