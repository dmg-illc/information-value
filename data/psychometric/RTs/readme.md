## Format
DataFrames are saved with a number of reading time columns from the *Revisiting UID* script:
 
```
[time_sum_mean,	time_sum_list,	time_mean,	time_count_nonzero, time_sum_mean_NO,	time_sum_list_NO,	time_count_nonzero_NO,	time_mean_N,]
```
- `time_sum_mean`: mean (sum of all word RTs in a sentence) across subjects
- `time_sum_list`: list of (sum of all word RTs in a sentence) per subject
- `time_mean`: mean of all word RTs in a sentence
- `time_count_nonzero`: mode of the number of words with non-zero RTs

`[..]_NO`: corresponds to No Outliers (they filter away any sentences that contain 'outlier' words (where outliers are any word with a z-score > 3 when the distribution of reading times is modeled as log-linear))
`df_full <- filter(agg_per_subject_sentence_full, agg_per_subject_sentence_full$outlier_sum==0)`

**TL;DR** use time_sum_list_NO
