library(data.table)

args = commandArgs(trailingOnly=TRUE)

results_dir <- args[1]

METADATA   <- paste0(results_dir, "/metadata.minimal.tsv")
NEXTCLADE  <- paste0(results_dir, "/nextclade.minimal.tsv")
OUTPUT     <- paste0(results_dir, "/minimal.tsv")

print(paste("Loading metadata:", METADATA))
metadata_df <- data.table::fread(METADATA)

print(paste("Loading nextclade:", NEXTCLADE))
nextclade_df <- data.table::fread(NEXTCLADE)

print(paste("Merging metadata and nextclade"))
meta_gisaid <- merge(metadata_df, nextclade_df, by=c("strain"), all.x=TRUE)

print(paste("Saving metadata:", OUTPUT))
data.table::fwrite(meta_gisaid, file=OUTPUT, sep="\t")

print("Done")
