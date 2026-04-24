import scanpy as sc
import pandas as pd
#from openai import OpenAI
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import celltypist
from celltypist import models
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
adata = sc.datasets.pbmc3k()

# Print basic info
print(adata)

# Identify mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')

#  Compute QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)


# Visualize QC metrics
sc.pl.violin(
    adata,
    ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    multi_panel=True
)


# Filter cells & genes

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Filter high mitochondrial cells
adata = adata[adata.obs.pct_counts_mt < 5, :]

# Print after filtering
print(adata)

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Keep only those genes
adata = adata[:, adata.var.highly_variable]

sc.pl.highly_variable_genes(adata)

# Save log-normalized data before scaling (CellTypist needs this)
adata.raw = adata
sc.pp.scale(adata, max_value=10)

# PCA
sc.tl.pca(adata)

# Neighborhood graph
sc.pp.neighbors(adata)

# Clustering
sc.tl.leiden(adata)

# UMAP visualization
sc.tl.umap(adata)

# Plot
sc.pl.umap(adata, color=['leiden'])


# Download immune model (only downloads once, ~50MB)
models.download_models(model='Immune_All_Low.pkl')

# CellTypist needs log-normalized data — you already have this
# But it needs raw counts in adata.X for its internal check, so we annotate directly
predictions = celltypist.annotate(
    adata.raw.to_adata(),
    model='Immune_All_Low.pkl',
    majority_voting=True
)

# Add ground truth labels to adata
adata.obs['celltypist_label'] = predictions.predicted_labels['majority_voting']

# Visualize to verify it makes sense
sc.pl.umap(adata, color=['celltypist_label'])

print(adata.obs['celltypist_label'].value_counts())

sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')

sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False)



# Extract marker genes into a dataframe
markers = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10)

print(markers)

adata.obs['cell_type'] = adata.obs['leiden'].map({
    '0': 'T cells',
    '1': 'B cells',
    '2': 'Monocytes',
    '3': 'NK cells',
    '4': 'Dendritic cells'
})

sc.pl.umap(adata, color=['cell_type'])

cluster_markers = {}

# get cluster names
clusters = adata.uns['rank_genes_groups']['names'].dtype.names

# extract top 10 genes per cluster
for cluster in clusters:
    genes = adata.uns['rank_genes_groups']['names'][cluster][:10]
    cluster_markers[cluster] = list(genes)

print(cluster_markers)

gene_df = pd.read_csv("gene_database.csv")

# Combine into searchable text
gene_df["text"] = (
    gene_df["gene"] + ": " +
    gene_df["description"] + " (" +
    gene_df["cell_type"] + ")"
)

texts = gene_df["text"].tolist()


# EMBEDDINGS
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(texts)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


# RULE-BASED PREDICTION
def predict_cell_type(genes):
    monocyte_genes = {'LYZ', 'S100A8', 'S100A9', 'CST3', 'TYROBP', 'FCN1',
                      'AIF1', 'FCER1G', 'LST1', 'CD14', 'VCAN', 'FTL', 'FTH1',
                      'COTL1', 'SAT1', 'S100A6'}
    tcell_genes = {'CD3D', 'CD3E', 'CD3G', 'IL7R', 'TCF7', 'CD4', 'CD8A',
                   'IL32', 'CD2', 'CD7', 'TRAC', 'TRBC1', 'LTB', 'LDHB'}
    bcell_genes = {'MS4A1', 'CD79A', 'CD79B', 'CD19', 'BANK1', 'IGHM',
                   'IGKC', 'CD22', 'FCRLA'}
    nk_genes = {'GNLY', 'NKG7', 'GZMB', 'PRF1', 'KLRD1', 'NCAM1',
                'KLRB1', 'KLRC1', 'GZMA', 'FGFBP2'}
    platelet_genes = {'PPBP', 'PF4', 'GP1BA', 'ITGA2B', 'TUBB1', 'NRGN', 'GNG11'}
    dendritic_genes = {'FCER1A', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRA', 'HLA-DRB1',
                       'CD1C', 'CLEC10A', 'CLEC9A', 'LILRA4', 'IRF7', 'HLA-DPB1'}
    proliferating_genes = {'KIAA0101', 'MKI67', 'TOP2A', 'PCNA', 'ZWINT',
                           'CDK1', 'CCNB1', 'STMN1'}

    gene_set = set(genes)

    scores = {
        'T cells': len(gene_set & tcell_genes),
        'B cells': len(gene_set & bcell_genes),
        'NK cells': len(gene_set & nk_genes),
        'Dendritic cells': len(gene_set & dendritic_genes),
        'Monocytes': len(gene_set & monocyte_genes),
        'Platelets': len(gene_set & platelet_genes),
        'Proliferating cells': len(gene_set & proliferating_genes),
    }

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"


def retrieve_knowledge(genes, top_k=2):
    results = []

    for gene in genes[:5]:  # use top marker genes
        query_embedding = model.encode([gene])
        distances, indices = index.search(np.array(query_embedding), top_k)

        for idx in indices[0]:
            results.append(texts[idx])

    # remove duplicates + limit size
    results = list(set(results))
    return results[:10]


# LLM SETUP

client = Groq(api_key="GROQ_API_KEY")

def ask_llm(genes, knowledge):
    prompt = f"""
    You are an expert in single-cell RNA sequencing.

    Marker genes:
    {genes}

    Retrieved biological knowledge:
    {knowledge}

    Rules:
    - Only trust knowledge that directly matches the marker genes
    - Ignore unrelated knowledge
    - If knowledge is weak or conflicting, say "uncertain"
    - Do not guess

    Task:
    1. Identify cell type
    2. Explain reasoning
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

def compute_confidence(genes, knowledge_list):
    gene_set = set(genes)
    votes = {}

    for snippet in knowledge_list:
        # snippet looks like "LYZ: Monocyte marker (Monocytes)"
        # extract the cell type in parentheses
        if "(" in snippet and ")" in snippet:
            cell_type = snippet.split("(")[-1].replace(")", "").strip()
            votes[cell_type] = votes.get(cell_type, 0) + 1

    total = sum(votes.values())
    if total == 0:
        return {}, "No knowledge retrieved"

    confidence = {k: round(v/total * 100, 1) for k, v in votes.items()}
    top = max(confidence, key=confidence.get)
    return confidence, top

def parse_llm_prediction(llm_text):
    llm_text_lower = llm_text.lower()
    if "monocyte" in llm_text_lower:
        return "Monocytes"
    elif "t cell" in llm_text_lower or "t-cell" in llm_text_lower:
        return "T cells"
    elif "b cell" in llm_text_lower or "b-cell" in llm_text_lower:
        return "B cells"
    elif "nk cell" in llm_text_lower or "natural killer" in llm_text_lower:
        return "NK cells"
    elif "dendritic" in llm_text_lower:
        return "Dendritic cells"
    elif "platelet" in llm_text_lower or "megakaryocyte" in llm_text_lower:
        return "Platelets"
    elif "proliferat" in llm_text_lower:
        return "Proliferating cells"
    elif "uncertain" in llm_text_lower:
        return "Unknown"
    else:
        return "Unknown"


# RUN FOR SELECTED CLUSTERS

results = []

for cluster, genes in cluster_markers.items():

    # RAG
    info = retrieve_knowledge(genes)

    # Rule-based
    rule_prediction = predict_cell_type(genes)

    # LLM
    llm_response = ask_llm(genes, info)

    print(f"\nCluster {cluster}")
    print("Genes:", genes)
    print("Knowledge:", info)
    print("Rule-based:", rule_prediction)
    print("LLM Response:\n", llm_response)

    llm_prediction = parse_llm_prediction(llm_response)

    results.append({
        "cluster": cluster,
        "genes": genes,
        "knowledge": info,
        "rule_based": rule_prediction,
        "llm": llm_response,
        "llm_prediction": llm_prediction
    })
confidence_scores, confident_prediction = compute_confidence(genes, info)
print("Confidence scores:", confidence_scores)

# SAVE RESULTS

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)

print("\nSaved results to results.csv")



# CONFUSION MATRIX

ground_truth_map = (
    adata.obs.groupby('leiden')['celltypist_label']
    .agg(lambda x: x.value_counts().index[0])
    .to_dict()
)

label_map = {
    'Tcm/Naive helper T cells': 'T cells',
    'Tem/Trm cytotoxic T cells': 'T cells',
    'Tem/Effector helper T cells': 'T cells',
    'Double-positive thymocytes': 'T cells',
    'Classical monocytes': 'Monocytes',
    'Non-classical monocytes': 'Monocytes',
    'Myelocytes': 'Monocytes',
    'B cells': 'B cells',
    'NK cells': 'NK cells',
    'CD16+ NK cells': 'NK cells',
    'DC2': 'Dendritic cells',
    'Megakaryocytes/platelets': 'Platelets',
}

comparison = []
for r in results:
    cluster = r['cluster']
    raw_gt = ground_truth_map.get(cluster, 'Unknown')
    normalized_gt = label_map.get(raw_gt, raw_gt)
    comparison.append({
        'cluster': cluster,
        'ground_truth': normalized_gt,
        'rule_based': r['rule_based'],
        'llm': r['llm_prediction']        # now using parsed LLM label
    })

comp_df = pd.DataFrame(comparison)
print("\nCluster comparison:")
print(comp_df)

# Accuracy for both
rule_acc = accuracy_score(comp_df['ground_truth'], comp_df['rule_based'])
llm_acc = accuracy_score(comp_df['ground_truth'], comp_df['llm'])
print(f"\nRule-based accuracy: {rule_acc:.2%}")
print(f"LLM accuracy:        {llm_acc:.2%}")

# Side by side confusion matrices
labels = sorted(comp_df['ground_truth'].unique())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, col, title in zip(axes,
                           ['rule_based', 'llm'],
                           ['Rule-based', 'LLM (Groq/Llama)']):
    cm = confusion_matrix(comp_df['ground_truth'], comp_df[col], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels,
                cmap='Blues', ax=ax)
    ax.set_xlabel(f'Predicted ({title})')
    ax.set_ylabel('Ground Truth (CellTypist)')
    ax.set_title(f'{title} vs CellTypist')

plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")
print("\n=== FINAL SUMMARY ===")
print(f"Total clusters evaluated: {len(comp_df)}")
print(f"Rule-based accuracy: {rule_acc:.2%}")
print(f"LLM accuracy:        {llm_acc:.2%}")
print(f"\nRule-based errors:")
print(comp_df[comp_df['ground_truth'] != comp_df['rule_based']][['cluster','ground_truth','rule_based']])
print(f"\nLLM errors:")
print(comp_df[comp_df['ground_truth'] != comp_df['llm']][['cluster','ground_truth','llm']])