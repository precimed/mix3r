library(eulerr)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)
# args[1] : path to *.parameters.csv file produced by extract_p.py; "bdbiobank_bdclinical_bdselfreport_sep12.parameters.csv"
# args[2] : path to output file; "bdbiobank_bdclinical_bdselfreport_sep12.parameters.euler.png"
# args[3], args[4], args[5] : labels for trait 1, trait 2 and trait 3 in Euler diagram; "BD biobank" "BD clinical" "BD self-repotr"
# args[6], args[7], args[8] : hex colors for trait 1, trait 2 and trait 3 in Euler diagram; "#77AADD" "#EE8866" "#EEDD88"

if (length(args) < 5) {
  stop("Missing arguments!")
} else {
  fname <- args[1]
  outf <- args[2]
  labels <- c(args[3], args[4], args[5])
  if (length(args) == 8) {
    colors <- c(args[6], args[7], args[8])
  } else {
    colors <- c("#77AADD", "#EE8866", "#EEDD88")
  }
} 

df <- fread(fname)
i_best_run = which.min(df$rank_p_proportion_deviation_from_median)
best = df[i_best_run]

acceptable_negative_gap = -1E-6
p1 <- best$p_1 - best$p_12 - best$p_13 + best$p_123
if (p1<0 & p1>acceptable_negative_gap) {p1 <- 0}
p2 <- best$p_2 - best$p_12 - best$p_23 + best$p_123
if (p2<0 & p2>acceptable_negative_gap) {p2 <- 0}
p3 <- best$p_3 - best$p_13 - best$p_23 + best$p_123
if (p3<0 & p3>acceptable_negative_gap) {p3 <- 0}
p12 <- best$p_12 - best$p_123
if (p12<0 & p12>acceptable_negative_gap) {p12 <- 0}
p13 <- best$p_13 - best$p_123
if (p13<0 & p13>acceptable_negative_gap) {p13 <- 0}
p23 <- best$p_23 - best$p_123
if (p23<0 & p23>acceptable_negative_gap) {p23 <- 0}
p123 <- best$p_123
if (p123<0 & p1>acceptable_negative_gap) {p123 <- 0}
factor = 10 # otherwise behaves badly with small numbers
vec2plot = c("1"=factor*p1,"2"=factor*p2,"3"=factor*p3,"1&2"=factor*p12,"1&3"=factor*p13,"2&3"=factor*p23,"1&2&3"=factor*p123)

fit <- euler(vec2plot, input="disjoint", shape="ellipse")

edge_lty = 1:1:1
alpha = 1
label_color = "black"
edge_color = "white"
label_fontsize = 24
quantile_fontsize = 24
edge_width = 4

png(filename=paste0(outf,".png"), width=600, height=600, units="px", pointsize=12, bg="white", res=NA)
plot(fit,
     fills = list(fill=colors, alpha=alpha),
     labels = list(labels=labels, col=label_color, fontsize=label_fontsize),
     edges = list(col=edge_color, lex=edge_width),
     lty = edge_lty,
     quantities = list(type="percent", cex=1, fontsize=quantile_fontsize))
supressed_output = dev.off()

svg(filename=paste0(outf,".svg"), bg="white", onefile=TRUE)
plot(fit,
     fills = list(fill=colors, alpha=alpha),
     labels = list(labels=labels, col=label_color, fontsize=label_fontsize),
     edges = list(col=edge_color, lex=edge_width),
     lty = edge_lty,
     quantities = list(type="percent", cex=1, fontsize=quantile_fontsize))
supressed_output = dev.off()

cat(paste0("Figure saved to: ", outf, ".png", "\n"))
cat(paste0("Figure saved to: ", outf, ".svg", "\n"))
