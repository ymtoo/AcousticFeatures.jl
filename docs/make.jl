using Documenter

push!(LOAD_PATH,"../src/")
using AcousticFeatures

makedocs(
    sitename = "AcousticFeatures.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "index.md",
        "acousticfeatures.md",
    ],
)

deploydocs(
  repo = "github.com/ymtoo/AcousticFeatures.jl",
  versions = ["dev" => "master"],
  branch = "gh-pages",
)