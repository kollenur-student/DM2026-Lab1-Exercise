import os
import tempfile
from typing import Dict, List, Any, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI modules
import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpg
import PAMI.frequentPattern.topk.FAE as topk_fae
import PAMI.frequentPattern.maximal.MaxFPGrowth as max_fpg


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = 1,
        k: Optional[int] = 1,
    ) -> Dict[str, Any]:
        """Mine frequent word patterns for a specific document category using PAMI.

        Args:
            category_name: Category label string to mine patterns for.
            filtering_method: Vocabulary filter method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold required for 'fpgrowth' and 'maxfpgrowth' (default 1).
            k: Top-k pattern count required for 'topk' algorithm (default 1).

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, n_patterns, and patterns list.
        """
        result_id = session.next_result_id("mine_patterns")

        if session.dataframe is None:
            summary = {"result_id": result_id, "error": "No DataFrame is loaded in session state."}
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        df = session.dataframe

        cat_col = "category_name" if "category_name" in df.columns else "label"
        if cat_col not in df.columns:
            summary = {"result_id": result_id, "error": f"Category column '{cat_col}' not found in DataFrame."}
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        # 1. Filter DataFrame to target category
        cat_df = df[df[cat_col].astype(str) == str(category_name)].copy()
        if cat_df.empty:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "n_patterns": 0,
                "patterns": [],
                "note": f"No documents found for category '{category_name}'."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 2. Build category-local DTM with stop_words="english"
        texts = cat_df["text"].fillna("").astype(str).tolist()
        vectorizer = CountVectorizer(stop_words="english")
        dtm_sparse = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        if len(feature_names) == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "n_patterns": 0,
                "patterns": [],
                "note": "No vocabulary terms survived English stopword removal."
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dtm_dense = dtm_sparse.toarray()
        dtm_df = pd.DataFrame(dtm_dense, columns=feature_names)

        # 3. Apply vocabulary filtering method
        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                scores = dtm_df.var(axis=0, ddof=0)
            else:
                scores = dtm_df.sum(axis=0)

            # Keep middle 90% (5th to 95th percentile)
            p_low = np.percentile(scores, 5)
            p_high = np.percentile(scores, 95)
            keep_mask = (scores > p_low) & (scores < p_high)
            filtered_df = dtm_df.loc[:, keep_mask]
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm_sparse).toarray()
            mean_tfidf = tfidf.mean(axis=0)
            # Keep top 80% (>= 20th percentile)
            p_20 = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= p_20
            filtered_df = dtm_df.loc[:, keep_mask]
        else:
            filtered_df = dtm_df

        # Edge case: No terms survived filtering
        if filtered_df.shape[1] == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 4. Convert filtered DataFrame to transactional database file for PAMI
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "transactions.txt")
            converter = db_converter.DF2DB(filtered_df)
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            # Mine patterns using selected algorithm
            if algorithm == "topk":
                target_k = k if k is not None else 1
                miner = topk_fae.FAE(iFile=db_file, k=target_k, sep="\t")
            elif algorithm == "maxfpgrowth":
                target_sup = min_sup if min_sup is not None else 1
                miner = max_fpg.MaxFPGrowth(iFile=db_file, minSup=target_sup, sep="\t")
            else:  # default fpgrowth
                target_sup = min_sup if min_sup is not None else 1
                miner = fpg.FPGrowth(iFile=db_file, minSup=target_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

        # Parse extracted patterns
        mined_patterns = []
        if patterns_df is not None and not patterns_df.empty:
            for _, row in patterns_df.iterrows():
                p_items = [item.strip() for item in str(row["Patterns"]).split("\t") if item.strip()]
                p_sup = int(row["Support"])
                mined_patterns.append({"pattern": p_items, "support": p_sup})

        # Save artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = mined_patterns

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(mined_patterns),
            "patterns": mined_patterns,
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )

        return summary

    return [mine_patterns_tool]
