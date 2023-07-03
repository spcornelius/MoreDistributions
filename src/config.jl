const DEFAULT_NLOPT_OPTIONS = Dict(
    "xtol_rel" => 1e-8,
    "ftol_rel" => 1e-8,
    "initial_step" => 1e-6
)

const DEFAULT_NLOPT_ALG = :LN_NEWUOA_BOUND