# Scaled coalescent and stepwise-mutation simulator for the ABC lab.
# n sampled Y chromosomes share one ancestry tree across all loci.

scaled_hazard <- function(u, beta, gamma) {
  stopifnot(all(is.finite(u)), all(u >= 0),
            is.finite(beta), beta >= 0, is.finite(gamma), gamma >= 0)
  if (beta == 0) {
    stopifnot(gamma == 0)
    return(u)
  }
  tau <- gamma / beta
  end_hazard <- -expm1(-gamma) / beta
  ans <- numeric(length(u))
  growing <- u <= tau
  small <- growing & beta * u < 700
  ans[small] <- exp(-gamma) * expm1(beta * u[small]) / beta
  large <- growing & !small
  ans[large] <- (exp(beta * u[large] - gamma) - exp(-gamma)) / beta
  ans[!growing] <- end_hazard + u[!growing] - tau
  ans
}

inverse_scaled_hazard <- function(h, beta, gamma) {
  stopifnot(all(is.finite(h)), all(h >= 0),
            is.finite(beta), beta >= 0, is.finite(gamma), gamma >= 0)
  if (beta == 0) {
    stopifnot(gamma == 0)
    return(h)
  }
  tau <- gamma / beta
  end_hazard <- -expm1(-gamma) / beta
  ans <- numeric(length(h))
  growing <- h <= end_hazard & h > 0
  z <- log(beta) + log(h[growing]) + gamma
  # softplus(z) = log(1 + exp(z)), without overflow for large growth.
  ans[growing] <- ifelse(z > 35, z + log1p(exp(-z)),
                         log1p(exp(pmin(z, 35)))) / beta
  ans[h > end_hazard] <- tau + h[h > end_hazard] - end_hazard
  ans
}

# `root` is the repeat count of the sample's common ancestor at each locus.
# The summaries below do not depend on it; it only shifts each column.
simulate_data <- function(theta, beta, gamma, n, loci = 8L,
                          root = rep(0L, loci),
                          return_genealogy = FALSE) {
  stopifnot(length(theta) == 1L, is.finite(theta), theta >= 0,
            length(beta) == 1L, is.finite(beta), beta >= 0,
            length(gamma) == 1L, is.finite(gamma), gamma >= 0,
            length(n) == 1L, n >= 2L, n == as.integer(n),
            length(loci) == 1L, loci >= 1L, loci == as.integer(loci),
            length(root) == loci, all(is.finite(root)),
            all(root == round(root)))
  n <- as.integer(n)
  loci <- as.integer(loci)
  nodes <- 2L * n - 1L
  time <- numeric(nodes)
  child1 <- child2 <- integer(nodes)
  active <- seq_len(n)
  hazard_clock <- 0

  for (parent in (n + 1L):nodes) {
    k <- length(active)
    hazard_clock <- hazard_clock + rexp(1L, rate = k * (k - 1) / 2)
    time[parent] <- inverse_scaled_hazard(hazard_clock, beta, gamma)
    pair <- sample.int(k, 2L)
    child1[parent] <- active[pair[1L]]
    child2[parent] <- active[pair[2L]]
    active <- c(active[-pair], parent)
  }

  states <- matrix(0, nrow = nodes, ncol = loci)
  states[nodes, ] <- root
  for (parent in nodes:(n + 1L)) {
    for (child in c(child1[parent], child2[parent])) {
      branch_length <- time[parent] - time[child]
      stopifnot(branch_length >= 0)
      half_mean <- theta * branch_length / 4
      states[child, ] <- states[parent, ] +
        rpois(loci, half_mean) - rpois(loci, half_mean)
    }
  }
  observed <- states[seq_len(n), , drop = FALSE]
  storage.mode(observed) <- "integer"
  colnames(observed) <- if (is.null(names(root))) {
    paste0("marker", seq_len(loci))
  } else {
    names(root)
  }
  if (!return_genealogy) return(observed)

  edges <- data.frame(
    parent = rep((n + 1L):nodes, each = 2L),
    child = as.vector(t(cbind(child1[(n + 1L):nodes],
                              child2[(n + 1L):nodes]))))
  edges$parent_time <- time[edges$parent]
  edges$child_time <- time[edges$child]
  edges$length <- edges$parent_time - edges$child_time
  list(data = observed, edges = edges, root = nodes,
       node_time = time, node_states = states)
}

# Convert the genealogy returned by simulate_data(..., return_genealogy = TRUE)
# to an ape "phylo" object, e.g. for plot(ladderize(as_phylo(g))).
# Building the object needs only base R; plotting it needs the ape package.
as_phylo <- function(genealogy) {
  n <- nrow(genealogy$data)
  # ape numbers tips 1..n and internal nodes n+1..2n-1 with the root at n+1.
  # The simulator numbers internal nodes in merger order with the root last.
  relabel <- function(v) ifelse(v <= n, v, 3L * n - v)
  e <- genealogy$edges
  tree <- list(edge = cbind(relabel(e$parent), relabel(e$child)),
               edge.length = e$length,
               tip.label = paste0("chr", seq_len(n)),
               Nnode = n - 1L)
  storage.mode(tree$edge) <- "integer"
  class(tree) <- "phylo"
  tree
}

summary_stats <- function(x) {
  stopifnot(is.matrix(x), nrow(x) >= 2L, ncol(x) >= 1L,
            all(is.finite(x)), all(x == round(x)))
  diversity <- vapply(seq_len(ncol(x)), function(j) {
    p <- prop.table(table(x[, j]))
    1 - sum(p^2)
  }, 0.0)
  c(mean_variance = mean(apply(x, 2L, var)),
    mean_diversity = mean(diversity),
    distinct_profiles = nrow(unique(as.data.frame(x))))
}

# Beaumont et al. (2002), p. 2033: their cleaned real-data comparison uses
# mu ~ Gamma(shape 10, scale 0.00008), log(N_A) ~ N(8.5, 2^2),
# r ~ Exponential(mean 0.005), and t_g/N_A ~ Gamma(shape 2, scale 1).
# Draw them jointly and transform; the scaled components are dependent.
sample_real_prior <- function(B = 1L, include_natural = FALSE) {
  stopifnot(length(B) == 1L, B >= 1L, B == as.integer(B))
  mu <- rgamma(B, shape = 10, scale = 0.00008)
  ancestral_size <- rlnorm(B, meanlog = 8.5, sdlog = 2)
  growth_rate <- rexp(B, rate = 1 / 0.005)
  duration_ratio <- rgamma(B, shape = 2, scale = 1)
  beta <- growth_rate * ancestral_size
  scaled <- data.frame(theta = 2 * ancestral_size * mu,
                       beta = beta,
                       gamma = beta * duration_ratio)
  if (!include_natural) return(scaled)
  cbind(scaled, mu = mu, N_A = ancestral_size, r = growth_rate,
        duration_ratio = duration_ratio)
}

# Separate synthetic experiment; not the default prior for the real data.
sample_synthetic_prior <- function(B = 1L) {
  data.frame(theta = rgamma(B, shape = 4, scale = 1),
             beta = rgamma(B, shape = 3, scale = 2),
             gamma = rgamma(B, shape = 3, scale = 1))
}
