process plot_results {
    conda = params.condaEnv
    label 'small'

    publishDir params.dumpDir, mode: 'copy', overwrite: true

    input:
    path result_pickles

    output:
    path "*_best_stats.csv"
    path "*_all_cv_losses.png"

    script:
    """
    python -m fennec.plotter plot -- ${result_pickles}
    """
}
