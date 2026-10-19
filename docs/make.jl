using Documenter, ThreeDGaussianSplatting

PAGE_NAMES_AND_FILES = [
    "Home" => "index.md",
    "Mathematics" => "math.md",
    "Reference" => "reference.md"
]

makedocs(
    sitename = "ThreeDGaussianSplatting.jl",
    pages = PAGE_NAMES_AND_FILES,
    format = Documenter.HTML(;
        mathengine = Documenter.KaTeX(
            Dict(
                :macros => Dict(
                    raw"\vect" => raw"\boldsymbol{#1}",
                    raw"\matr" => raw"\boldsymbol{#1}",
                    raw"\norm" => raw"\left\| #1 \right\|",
                    raw"\abs" => raw"\left| #1 \right|",
                    raw"\pdiff" => raw"\frac{\partial #1}{\partial #2}",
                    raw"\SSIM" => raw"\operatorname{SSIM}",
                    raw"\DSSIM" => raw"\operatorname{DSSIM}",
                    raw"\ReLU" => raw"\operatorname{ReLU}",
                )
            )
        ),
        size_threshold = nothing
    ),
    remotes = nothing
)


#=
deploydocs(
    repo = "github.com/LaurensDiels/ThreeDGaussianSplatting.jl.git",
)
=#