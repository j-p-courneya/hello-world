"""
VisiumHD Pipeline
-----------------
End-to-end analysis of 10X Genomics Visium HD spatial transcriptomics data
using Scanpy and Squidpy.

Usage:
    from pipeline import VisiumHDPipeline

    pipe = VisiumHDPipeline(
        data_dir="/path/to/spaceranger/output",
        bin_size=8,
        output_dir="./results",
    )
    adata = pipe.run()
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=100, facecolor="white")


class VisiumHDPipeline:
    """
    Modular pipeline for 10X Visium HD spatial transcriptomics analysis.

    Parameters
    ----------
    data_dir : str
        Path to the Space Ranger output directory.
    bin_size : int
        Bin size in microns. One of {2, 8, 16}. Default is 8.
    output_dir : str
        Directory where results (plots, h5ad files) will be saved.
    """

    BIN_SIZES = {2, 8, 16}

    def __init__(
        self,
        data_dir: str,
        bin_size: int = 8,
        output_dir: str = "./results",
    ):
        if bin_size not in self.BIN_SIZES:
            raise ValueError(f"bin_size must be one of {self.BIN_SIZES}, got {bin_size}")

        self.data_dir = Path(data_dir)
        self.bin_size = bin_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[VisiumHDPipeline] data_dir  : {self.data_dir}")
        print(f"[VisiumHDPipeline] bin_size  : {self.bin_size} μm")
        print(f"[VisiumHDPipeline] output_dir: {self.output_dir}")

    # ------------------------------------------------------------------
    # STEP 1 — Load
    # ------------------------------------------------------------------
    def load(self) -> sc.AnnData:
        """
        Load Visium HD data from Space Ranger output.

        Visium HD stores binned outputs under:
            <data_dir>/binned_outputs/square_<NNN>um/
        """
        bin_dir = self.data_dir / f"binned_outputs/square_{self.bin_size:03d}um"

        if not bin_dir.exists():
            # Fallback: some Space Ranger versions use the root directly
            bin_dir = self.data_dir

        print(f"\n[1/8] Loading data from {bin_dir}")
        adata = sc.read_visium(bin_dir, count_file="filtered_feature_bc_matrix.h5")
        adata.var_names_make_unique()

        # Store raw counts before any modification
        adata.layers["counts"] = adata.X.copy()

        print(f"      Loaded {adata.n_obs} spots × {adata.n_vars} genes")
        return adata

    # ------------------------------------------------------------------
    # STEP 2 — Quality Control
    # ------------------------------------------------------------------
    def qc(
        self,
        adata: sc.AnnData,
        min_counts: int = 100,
        min_genes: int = 50,
        max_mt_pct: float = 20.0,
    ) -> sc.AnnData:
        """
        Compute and apply QC filters.

        Parameters
        ----------
        min_counts : int
            Minimum UMI counts per spot.
        min_genes : int
            Minimum number of detected genes per spot.
        max_mt_pct : float
            Maximum mitochondrial gene percentage.
        """
        print(f"\n[2/8] Quality control (min_counts={min_counts}, min_genes={min_genes}, max_mt%={max_mt_pct})")

        # Annotate mitochondrial genes (human: MT-, mouse: mt-)
        adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))

        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

        # Plot QC distributions before filtering
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        axes[0].hist(adata.obs["total_counts"], bins=60, color="#58a6ff", edgecolor="none")
        axes[0].set_xlabel("Total counts"); axes[0].set_title("UMI counts per spot")
        axes[1].hist(adata.obs["n_genes_by_counts"], bins=60, color="#3fb950", edgecolor="none")
        axes[1].set_xlabel("Genes detected"); axes[1].set_title("Genes per spot")
        axes[2].hist(adata.obs["pct_counts_mt"], bins=60, color="#f78166", edgecolor="none")
        axes[2].set_xlabel("MT %"); axes[2].set_title("Mitochondrial %")
        for ax in axes:
            ax.set_facecolor("#161b22"); fig.patch.set_facecolor("#0d1117")
            ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
            ax.title.set_color("white")
        plt.tight_layout()
        plt.savefig(self.output_dir / "qc_distributions.png", dpi=150)
        plt.close()

        # Filter spots
        n_before = adata.n_obs
        sc.pp.filter_cells(adata, min_counts=min_counts)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct].copy()

        # Filter lowly expressed genes (expressed in >= 10 spots)
        sc.pp.filter_genes(adata, min_cells=10)

        print(f"      Retained {adata.n_obs}/{n_before} spots after QC")
        return adata

    # ------------------------------------------------------------------
    # STEP 3 — Normalization
    # ------------------------------------------------------------------
    def normalize(self, adata: sc.AnnData) -> sc.AnnData:
        """Normalize to 10k counts per spot and log1p-transform."""
        print("\n[3/8] Normalizing")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    # ------------------------------------------------------------------
    # STEP 4 — Feature Selection
    # ------------------------------------------------------------------
    def select_features(
        self,
        adata: sc.AnnData,
        n_top_genes: int = 3000,
    ) -> sc.AnnData:
        """Identify highly variable genes."""
        print(f"\n[4/8] Selecting top {n_top_genes} highly variable genes")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            layer="counts",
        )
        print(f"      {adata.var['highly_variable'].sum()} HVGs selected")
        return adata

    # ------------------------------------------------------------------
    # STEP 5 — Dimensionality Reduction
    # ------------------------------------------------------------------
    def reduce(
        self,
        adata: sc.AnnData,
        n_pcs: int = 50,
        n_neighbors: int = 15,
    ) -> sc.AnnData:
        """PCA, neighbor graph, and UMAP."""
        print(f"\n[5/8] Dimensionality reduction (PCA n_pcs={n_pcs}, neighbors={n_neighbors})")
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", use_highly_variable=True)
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        return adata

    # ------------------------------------------------------------------
    # STEP 6 — Clustering
    # ------------------------------------------------------------------
    def cluster(
        self,
        adata: sc.AnnData,
        resolution: float = 0.5,
    ) -> sc.AnnData:
        """Leiden clustering."""
        print(f"\n[6/8] Leiden clustering (resolution={resolution})")
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden")
        n_clusters = adata.obs["leiden"].nunique()
        print(f"      Found {n_clusters} clusters")

        # UMAP colored by cluster
        sc.pl.umap(
            adata,
            color=["leiden"],
            save=str(self.output_dir / "umap_leiden.png"),
            show=False,
        )
        return adata

    # ------------------------------------------------------------------
    # STEP 7 — Spatial Visualization
    # ------------------------------------------------------------------
    def plot_spatial(
        self,
        adata: sc.AnnData,
        color: str = "leiden",
        genes: list = None,
    ):
        """
        Plot clusters and/or gene expression on the H&E image.

        Parameters
        ----------
        color : str
            Observation key to color by (e.g. "leiden").
        genes : list of str, optional
            Genes to visualize spatially.
        """
        print(f"\n[7/8] Spatial visualization")

        # Compute spatial neighbors (used for downstream SVG analysis)
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

        # Spatial cluster plot
        sq.pl.spatial_scatter(
            adata,
            color=color,
            library_id=list(adata.uns["spatial"].keys())[0],
            figsize=(8, 8),
            save=str(self.output_dir / f"spatial_{color}.png"),
        )

        # Optional gene expression overlay
        if genes:
            valid = [g for g in genes if g in adata.var_names]
            if valid:
                sq.pl.spatial_scatter(
                    adata,
                    color=valid,
                    library_id=list(adata.uns["spatial"].keys())[0],
                    figsize=(6 * len(valid), 6),
                    save=str(self.output_dir / "spatial_genes.png"),
                )
            else:
                print(f"      Warning: none of {genes} found in var_names")

    # ------------------------------------------------------------------
    # STEP 8 — Spatially Variable Genes
    # ------------------------------------------------------------------
    def spatially_variable_genes(
        self,
        adata: sc.AnnData,
        n_jobs: int = 4,
    ) -> pd.DataFrame:
        """
        Compute Moran's I spatial autocorrelation for all genes.

        Returns a DataFrame sorted by Moran's I score.
        """
        print(f"\n[8/8] Computing spatially variable genes (Moran's I)")

        if "spatial_connectivities" not in adata.obsp:
            sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)

        sq.gr.spatial_autocorr(
            adata,
            mode="moran",
            n_jobs=n_jobs,
            n_perms=100,
        )

        svg_df = (
            adata.uns["moranI"]
            .sort_values("I", ascending=False)
        )
        svg_df.to_csv(self.output_dir / "spatially_variable_genes.csv")
        print(f"      Top 5 SVGs:\n{svg_df.head()}")
        return svg_df

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        min_counts: int = 100,
        min_genes: int = 50,
        max_mt_pct: float = 20.0,
        n_top_genes: int = 3000,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        resolution: float = 0.5,
        marker_genes: list = None,
    ) -> sc.AnnData:
        """
        Execute all pipeline steps sequentially.

        Parameters
        ----------
        marker_genes : list of str, optional
            Genes to overlay on the spatial plot.

        Returns
        -------
        adata : AnnData
            Fully processed AnnData object.
        """
        adata = self.load()
        adata = self.qc(adata, min_counts=min_counts, min_genes=min_genes, max_mt_pct=max_mt_pct)
        adata = self.normalize(adata)
        adata = self.select_features(adata, n_top_genes=n_top_genes)
        adata = self.reduce(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
        adata = self.cluster(adata, resolution=resolution)
        self.plot_spatial(adata, color="leiden", genes=marker_genes)
        self.spatially_variable_genes(adata)

        # Save processed object
        out_h5ad = self.output_dir / "visiumhd_processed.h5ad"
        adata.write_h5ad(out_h5ad)
        print(f"\n[Done] Saved processed AnnData to {out_h5ad}")

        return adata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisiumHD Scanpy Pipeline")
    parser.add_argument("data_dir", help="Space Ranger output directory")
    parser.add_argument("--bin-size", type=int, default=8, choices=[2, 8, 16])
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--min-counts", type=int, default=100)
    parser.add_argument("--max-mt-pct", type=float, default=20.0)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument(
        "--marker-genes",
        nargs="*",
        default=None,
        help="Genes to overlay on spatial plot",
    )
    args = parser.parse_args()

    pipe = VisiumHDPipeline(
        data_dir=args.data_dir,
        bin_size=args.bin_size,
        output_dir=args.output_dir,
    )
    pipe.run(
        min_counts=args.min_counts,
        max_mt_pct=args.max_mt_pct,
        n_top_genes=args.n_top_genes,
        resolution=args.resolution,
        marker_genes=args.marker_genes,
    )
