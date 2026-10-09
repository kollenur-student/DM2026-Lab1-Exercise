# four categories report

The dataset was loaded with the four categories `alt.atheism`, `soc.religion.christian`, `comp.graphics`, and `sci.med`. The document-length analysis shows substantial variation within every category, with many long-document outliers. The category box plot was saved as `[exact plot filename]` and corresponds to result `[describe_data_result_id]`.

A document-term matrix was created and inspected using a heatmap of selected documents and terms. Most entries are zero, while a small number of terms occur repeatedly in individual documents. The heatmap is saved as `[exact DTM heatmap filename]` and corresponds to result `[dtm_heatmap_result_id]`.

Feature relationships were examined with a correlation heatmap. Several terms show strong positive correlations, indicating that they frequently occur together, while other terms have weak or near-zero relationships. The correlation visualization is saved as `[exact correlation plot filename]` and corresponds to result `[feature_correlation_result_id]`.

Finally, the documents were projected into two dimensions using t-SNE. The resulting plot shows partial grouping by category, although the classes overlap considerably in the central region. This suggests that the document-term representation contains category-related structure but does not separate all categories perfectly. The plot is saved as `[exact dimension-reduction plot filename]` and corresponds to result `[reduce_dimensions_result_id]`.
