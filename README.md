# Information Value
This is the repository for the EMNLP 2023 paper: 

**Information Value: Measuring Utterance Predictability as Distance from Plausible Alternatives**  
Mario Giulianelli, Sarenne Wallbridge, and Raquel Fernández  

The 1.3M LM-generated alternatives used in this paper can be downloaded [here](https://doi.org/10.5281/zenodo.10006413).

---

### How to reproduce our analyses 
We provide the code for each set of experiments in the paper in `code/notebooks`. Predictability estimates, obtained using both information value and surprisal, are in the `data` folder. Note that we often refer to information value as 'surprise'.



### How to reproduce our experimental setup from scratch
- *Fine-tune LMs on dialogue*. First, preprocess the dialogue datasets using `code/switchboard_to_txt.py` and `code/dailydialog_to_txt.py`. Then, fine-tune our selection of autoregressive pre-trained language model using `code/run_clm.py`.
- *Pre-process acceptability judgements*. Use `code/to_jsonl.py` to convert the Clasp, Switchboard, and Dailydialog acceptability judgements from csv to jsonl. The reading times dataset are already in the right format.
- *Generate alternatives*. Run `code/generate_alternatives.py` one dataset at a time. (The LM-generated alternatives we use in our experiments can be downloaded [here](https://doi.org/10.5281/zenodo.10006413).)
- *Compute predictability*. Compute predictability estimates using `code/compute_information_value.py` and `code/compute_surprisal.py`.

Now you can run our analyses in `code/notebooks`.



