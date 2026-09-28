# Simulator for the Ridgecrest model-checking lab: aftershocks in one band
# around one gently curved fault, plus a small amount of broadly scattered
# activity. The model and fitted values are described in 10-model-description.html.
#
# One simulated catalog, in this order:
#   1. N ~ Poisson(Lambda), Lambda = integral of K (t + c)^(-p) over [t_start, t_end)
#   2. times: N iid draws from the density proportional to (t + c)^(-p) on
#      [t_start, t_end), by inverting its CDF; then sorted
#   3. magnitudes: M = m_min + Exponential(rate = b * log(10)), iid
#   4. scatter indicator S ~ Bernoulli(eps), iid
#   5. along-fault position U:
#        S = 0 (band):    U ~ g, the fitted smoothed histogram (pick a 1-km bin
#                         with its probability, then a uniform point inside it)
#        S = 1 (scatter): U ~ Uniform(u_lo, u_hi)
#   6. across-fault position V:
#        S = 0 (band):    V ~ Normal(m(U), sigma_v^2): centred on the fault
#                         curve m, with the same spread wherever U is
#        S = 1 (scatter): V ~ Uniform(-v_scatter, v_scatter)
#   7. apply the same observation window as the observed catalog
# Events are independent; for each event, time, magnitude and location are
# independent.

fit <- list(
  t_start = 0.25, t_end = 30,      # days after the M7.1 mainshock (fixed design)
  m_min = 2.5,                     # magnitude threshold (fixed design)
  K = 268.7, c = 0.01, p = 1.160,  # Omori-Utsu rate, events/day; c is fixed, not fitted
  b = 1.069,                       # Gutenberg-Richter b-value
  g = read.csv("data/along_fault_density.csv"),  # 1-km bins: u_left, u_right, u_mid,
                                   # prob (g) and centre_km (the fault curve m at u_mid)
  sigma_v = 1.86,                  # width of the band around the fault curve, km
  eps = 0.069,                     # share of broadly scattered events
  u_lo = -33.66, u_hi = 23.40,     # along-fault extent of the scatter, km
  v_scatter = 12                   # scatter fills |v| <= 12 km
)

# Fixed observation window in (U, V) coordinates (km), shared with the data
window <- list(u = c(-42, 26), v = c(-12, 12))

# The fault curve m(u): linear interpolation between 1-km bin centres
fault_centre <- function(u) approx(fit$g$u_mid, fit$g$centre_km, u, rule = 2)$y

# Signed distance across the fault from the curve (NE positive)
offset_from_fault <- function(cat) cat$v_km - fault_centre(cat$u_km)

apply_window <- function(cat) {
  keep <- cat$t_days >= fit$t_start & cat$t_days < fit$t_end &
    cat$mag >= fit$m_min &
    cat$u_km >= window$u[1] & cat$u_km <= window$u[2] &
    cat$v_km >= window$v[1] & cat$v_km <= window$v[2]
  cat[keep, ]
}

simulate_catalog <- function(fit) {
  # --- 1-2. times: nonhomogeneous Poisson process with Omori-Utsu rate
  A <- (fit$t_start + fit$c)^(1 - fit$p)
  B <- (fit$t_end + fit$c)^(1 - fit$p)
  expected_n <- fit$K * (B - A) / (1 - fit$p)
  n <- rpois(1, expected_n)
  t_days <- sort((A + runif(n) * (B - A))^(1 / (1 - fit$p)) - fit$c)

  # --- 3. magnitudes: Gutenberg-Richter above the threshold
  mag <- fit$m_min + rexp(n, rate = fit$b * log(10))

  # --- 4. which events are broadly scattered
  scatter <- runif(n) < fit$eps

  # --- 5. along-fault position
  # (ifelse draws a candidate from both branches and keeps the matching one)
  bin <- sample(nrow(fit$g), n, replace = TRUE, prob = fit$g$prob)
  u_km <- ifelse(scatter, runif(n, fit$u_lo, fit$u_hi),
                          fit$g$u_left[bin] + runif(n) * (fit$g$u_right[bin] - fit$g$u_left[bin]))

  # --- 6. across-fault position
  v_km <- ifelse(scatter, runif(n, -fit$v_scatter, fit$v_scatter),
                          fault_centre(u_km) + rnorm(n, 0, fit$sigma_v))

  # --- 7. same filters as the observed catalog
  apply_window(data.frame(t_days, mag, u_km, v_km))
}
