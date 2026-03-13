import pandas as pd

results = ["./mit-b3_adam_mit-b3_224_segmentation_ablation_study_results.csv", "./resnet_224_mit-b3_adam_simple_baseline_segmentation_ablation_study_results.csv", "./resnet_adam_simple_baseline_resnet_adam_baseline_segmentation_ablation_study_results.csv"]

df = pd.DataFrame()
for r in results:
    df1 = pd.read_csv(r)
    df = pd.concat([df, df1], ignore_index=True)

df.to_csv("./first_twenty_seg_results.csv", index=False)