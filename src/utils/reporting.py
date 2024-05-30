from statistics import fmean


def report_stats_original_datasets(dialogue_df, knowledge_df):
    print(f"Utterances in DIALOCONAN: {len(dialogue_df)}\n"
          f"Dialogues in DIALOCONAN: {len(dialogue_df['dialogue_id'].unique())}\n"
          f"HS-CN pairs in k-CONAN: {len(knowledge_df)}\n"
          f"\n"
          f"HS in DIALOCONAN: {len(dialogue_df['text'][dialogue_df['type'] == 'HS'])}\n"
          f"CN in DIALOCONAN: {len(dialogue_df['text'][dialogue_df['type'] == 'CN'])}\n"
          f"Unique HS in DIALOCONAN: {len(dialogue_df['text'][dialogue_df['type'] == 'HS'].unique())}\n"
          f"Unique CN in DIALOCONAN: {len(dialogue_df['text'][dialogue_df['type'] == 'CN'].unique())}\n"
          f"\n"
          f"Unique HS in k-CONAN: {len(knowledge_df['hate_speech'].unique())}\n"
          f"Unique CN in k-CONAN: {len(knowledge_df['counter_narrative'].unique())}\n"
          f"Unique knowledge in k-CONAN: {len(knowledge_df['knowledge_sentence'].unique())}\n"
          f"\n"
          f"Average utterances per dialogue in DIALOCONAN: {len(dialogue_df)} / {len(dialogue_df['dialogue_id'].unique())} = {len(dialogue_df) / len(dialogue_df['dialogue_id'].unique())}")


def report_stats_overlap_datasets(overlap_df, dialogue_df):
    print(f"Matched utterances in k-DIALOCONAN: {len(overlap_df[['dialogue_id', 'turn_id']].drop_duplicates())}\n")

    print(f"Unique dialogues with knowledge matches: {len(overlap_df['dialogue_id'].unique())}\n"
          f"Unique utterances with knowledge matches: {len(overlap_df['text'].unique())}\n"
          f"\n"
          f"Unique HS with relevant knowledge: {len(overlap_df['hate_speech'].unique())}\n"
          f"Unique CN: {len(overlap_df['text'][dialogue_df['type'] == 'CN'].unique())}\n"
          f"Unique knowledge sentences: {len(overlap_df['knowledge_sentence'].unique())}\n"
          f"Unique k-CN: {len(overlap_df['counter_narrative'].unique())}\n")


def report_stats_dataset(dataset, name=''):
    hs = [u['text'] for d in dataset for u in d['utterances'] if u['speaker'] == 'HS']
    cn = [u['text'] for d in dataset for u in d['utterances'] if u['speaker'] == 'CN']
    k = sum([u['knowledge']['knowledge_sentence'] for d in dataset for u in d['utterances']], [])
    kcn = sum([u['knowledge']['counter_narrative'] for d in dataset for u in d['utterances']], [])

    print(f"Dialogues in {name}: {len(dataset)}\n"
          f"Utterances in {name}: {sum([len(d['utterances']) for d in dataset])}\n"
          f"Utterances (HS) with knowledge in {name}: {sum([int(u['knowledge_grounded']) for d in dataset for u in d['utterances']])}\n"
          f"\n"
          f"HS in {name}: {sum([int(u['speaker'] == 'HS') for d in dataset for u in d['utterances']])}\n"
          f"CN in {name}: {sum([int(u['speaker'] == 'CN') for d in dataset for u in d['utterances']])}\n"
          f"Knowledge items in {name}: {sum([len(u['knowledge']['knowledge_sentence']) for d in dataset for u in d['utterances']])}\n"
          f"k-CN in {name}: {sum([len(u['knowledge']['counter_narrative']) for d in dataset for u in d['utterances']])}\n"
          f"\n"
          f"Unique HS in {name}: {len(set([u['text'] for d in dataset for u in d['utterances'] if u['speaker'] == 'HS']))}\n"
          f"Unique CN in {name}: {len(set([u['text'] for d in dataset for u in d['utterances'] if u['speaker'] == 'CN']))}\n"
          f"Unique knowledge items in {name}: {len(set(sum([u['knowledge']['knowledge_sentence'] for d in dataset for u in d['utterances']], [])))}\n"
          f"Unique k-CN in {name}: {len(set(sum([u['knowledge']['counter_narrative'] for d in dataset for u in d['utterances']], [])))}\n"
          f"\n"
          f"Average utterances per dialogue {name}: {fmean([len(d['utterances']) for d in dataset])}\n"
          f"Average utterances with knowledge per dialogue {name}: {fmean([len([ug for ug in d['utterances'] if ug['knowledge_grounded']]) for d in dataset])}\n"
          f"Average knowledge items per dialogue {name}: {fmean([sum([len(u['knowledge']['knowledge_sentence']) for u in d['utterances']]) for d in dataset])}\n"
          f"Average k-CN per dialogue {name}: {fmean([sum([len(u['knowledge']['counter_narrative']) for u in d['utterances']]) for d in dataset])}\n"
          f"\n"
          f"Average knowledge items per utterance in {name}: {fmean([len(u['knowledge']['knowledge_sentence']) for d in dataset for u in d['utterances']])}\n"
          f"Average k-CN per utterance in {name}: {fmean([len(u['knowledge']['counter_narrative']) for d in dataset for u in d['utterances']])}\n"
          f"\n"
          f"Average knowledge items per HS in {name}: {fmean([len(u['knowledge']['knowledge_sentence']) for d in dataset for u in d['utterances'] if u['speaker'] == 'HS'])}\n"
          f"Average k-CN per HS in {name}: {fmean([len(u['knowledge']['counter_narrative']) for d in dataset for u in d['utterances'] if u['speaker'] == 'HS'])}\n")

    return hs, cn, k, kcn
