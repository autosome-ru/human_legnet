library(dplyr)
library(ggplot2)

ds_path <- "HepG2_asb_analysis.tsv"
database <- "ADASTRA"

variants <- read.csv(ds_path, 
                     sep="\t",
                     header=TRUE)
 
variants$pred_pref_allele <- ifelse(variants$pred_score > 0, 
                                    'ref',
                                    'alt')
v <- variants %>% filter(fdr_comb_pval < 0.05) 
print(cor(v$score, v$pred_score))
print(fisher.test(v$pref_allele, v$pred_pref_allele),  workspace=2e12)


v <- v %>% filter(pred_pval < 0.05)
v <- v %>% filter(pred_pref_allele != 'no')
print(cor(v$score, v$pred_score))
print(fisher.test(v$pref_allele, v$pred_pref_allele))


variants$concordant <- ( (variants$score <0 & variants$pred_score < 0) | (variants$score>0 & variants$pred_score>0))
mask <- variants$fdr_comb_pval<0.05 & variants$pred_pval<0.05
conc <- variants$concordant[mask]



ggplot(variants, aes(x=pred_score,y=score)) +
  geom_point(data=subset(variants, 
                         fdr_comb_pval>=0.05 | pred_pval>=0.05),
             size=3,
             shape=21, 
             color='lightgrey', 
             fill='lightgrey') +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0) +
  geom_point(data=subset(variants, 
                         fdr_comb_pval<0.05 & pred_pval<0.05), 
             size=3, shape=21, color='black', mapping=aes(fill=!concordant))+
  scale_fill_manual(values = c("lightgreen", "pink"),
                    labels=c(paste("Сoncordant: ", sum(conc)), 
                             paste("Discordant:", sum(1-conc)))) +
  theme_bw() +
  theme(legend.position= c(.15, .90),
        #legend.text=element_text(size=3),
        legend.title=element_blank(),
        legend.key.size = unit(0.2, "cm")) +
  xlab('Model score') + ylab(paste0(database,' score'))

